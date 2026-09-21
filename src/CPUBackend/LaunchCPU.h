// CPU replacement for CUDA's <<<grid, block>>> launch syntax.
//
// cmake/translate_cuda_to_cpu.py rewrites
//     some_kernel<<<grid, block>>>(a, b)
// into
//     swaptube_cpu::launcher("some_kernel", some_kernel, grid, block)(a, b)
// and picks launcher_barrier() instead for kernels whose body contains
// __syncthreads. Only the <<<...>>> token is substituted, so the argument list
// is never parsed.
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include "cuda_runtime.h"

namespace swaptube_cpu {

// Worker pool sized to the machine's core count. Persistent, because a 1080p
// frame issues dozens of launches and thread creation would dominate.
void run_blocks(unsigned num_blocks, const std::function<void(unsigned, unsigned)>& chunk);
unsigned worker_count();

// Per-kernel timing, enabled with SWAPTUBE_PROFILE_KERNELS=1 and dumped at
// exit. This is how you find out which kernels are worth moving to the Metal
// backend, and how you measure what moving them bought.
bool profiling_enabled();
void record_kernel_time(const char* name, double seconds);

class ScopedKernelTimer {
public:
    explicit ScopedKernelTimer(const char* name)
        : name_(name), start_(name ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}

    ~ScopedKernelTimer() {
        if (!name_) return;
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_;
        record_kernel_time(name_, elapsed.count());
    }

private:
    const char* name_; // null when profiling is off, which makes this a no-op
    std::chrono::steady_clock::time_point start_;
};

inline dim3 unflatten(unsigned i, const dim3& extent) {
    return dim3(i % extent.x, (i / extent.x) % extent.y, i / (extent.x * extent.y));
}

// ------------------------------------------------------------- fast path
// Blocks run in parallel; the threads of a block run sequentially on one
// worker. Valid for every kernel that does not call __syncthreads.
template <class Kernel>
struct Launcher {
    const char* name;
    Kernel kernel;
    dim3 grid, block;

    template <class... Args>
    void operator()(Args&&... args) const {
        const unsigned nblocks = grid.x * grid.y * grid.z;
        if (nblocks == 0 || block.x * block.y * block.z == 0) return;
        ScopedKernelTimer timer(profiling_enabled() ? name : nullptr);
        const dim3 g = grid, b = block;
        Kernel k = kernel;

        run_blocks(nblocks, [&](unsigned begin, unsigned end) {
            tl_gridDim = g;
            tl_blockDim = b;
            for (unsigned bi = begin; bi < end; ++bi) {
                tl_blockIdx = unflatten(bi, g);
                for (unsigned tz = 0; tz < b.z; ++tz)
                    for (unsigned ty = 0; ty < b.y; ++ty)
                        for (unsigned tx = 0; tx < b.x; ++tx) {
                            tl_threadIdx = dim3(tx, ty, tz);
                            k(args...);
                        }
            }
        });
    }
};

// ---------------------------------------------------------- barrier path
// The threads of a block must make progress together, so they become real
// threads and blocks are processed one at a time. That keeps __shared__
// (a plain static) correct: never more than one block in flight.
void run_block_threads(unsigned num_blocks, const dim3& grid, const dim3& block,
                       const std::function<void()>& kernel_body);

template <class Kernel>
struct BarrierLauncher {
    const char* name;
    Kernel kernel;
    dim3 grid, block;

    template <class... Args>
    void operator()(Args&&... args) const {
        const unsigned nblocks = grid.x * grid.y * grid.z;
        if (nblocks == 0 || block.x * block.y * block.z == 0) return;
        ScopedKernelTimer timer(profiling_enabled() ? name : nullptr);
        Kernel k = kernel;
        run_block_threads(nblocks, grid, block, [&] { k(args...); });
    }
};

template <class Kernel>
Launcher<Kernel> launcher(const char* name, Kernel k, dim3 grid, dim3 block) { return {name, k, grid, block}; }

template <class Kernel>
BarrierLauncher<Kernel> launcher_barrier(const char* name, Kernel k, dim3 grid, dim3 block) { return {name, k, grid, block}; }

} // namespace swaptube_cpu
