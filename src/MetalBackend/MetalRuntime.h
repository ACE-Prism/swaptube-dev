// Host side of the Metal backend. Plain C++ interface; the implementation is
// Objective-C++ in MetalRuntime.mm.
//
// Objective-C++ rather than metal-cpp: metal-cpp would have to be vendored into
// the tree, while the Metal framework and its headers already ship with the
// Command Line Tools. One .mm file keeps the rest of the codebase plain C++.
#pragma once

#include <cstddef>
#include <cstdint>

namespace swaptube_metal {

// False when there is no Metal device or the shader library failed to compile,
// in which case every ported wrapper falls back to its CPU implementation.
bool available();

// Human-readable reason available() returned false, for the one-time warning.
const char* unavailable_reason();

// Device allocation. The buffers use shared storage, so the returned pointer is
// also valid on the CPU: a kernel that has not been ported yet can keep writing
// the same pixels with no copy in either direction.
void* allocate(size_t bytes);
void deallocate(void* pointer);

// True when the pointer came from allocate() and can therefore be bound to a
// kernel. Pointers from plain malloc cannot, which is what decides whether a
// wrapper is able to use the GPU for a given call.
bool is_device_pointer(const void* pointer);

// One dispatch of one kernel. Built per call; the expensive objects (device,
// queue, library, pipeline states) are cached behind the scenes.
class Dispatch {
public:
    explicit Dispatch(const char* kernel_name);
    ~Dispatch();

    Dispatch(const Dispatch&) = delete;
    Dispatch& operator=(const Dispatch&) = delete;

    // False when the kernel is missing from the library; callers fall back.
    bool valid() const;

    // The by-value arguments, as the struct generated into
    // src/Metal/generated/kernel_args.h. Always binds at index 0.
    void set_args(const void* args, size_t size);

    // A buffer previously returned by allocate(), bound at 1..n in the order the
    // pointer parameters appear in the CUDA signature.
    void set_buffer(unsigned index, const void* device_pointer);

    // Grid sizes in threads, not blocks: Metal's dispatchThreads handles a
    // non-multiple grid itself, so the kernels do not need bounds checks for
    // padding the way the CUDA versions do. They keep them anyway, harmlessly.
    void run_1d(unsigned count);
    void run_2d(unsigned width, unsigned height);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace swaptube_metal
