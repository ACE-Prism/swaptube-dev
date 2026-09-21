#include "LaunchCPU.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace swaptube_cpu {

thread_local dim3 tl_threadIdx;
thread_local dim3 tl_blockIdx;
thread_local dim3 tl_blockDim;
thread_local dim3 tl_gridDim;

namespace {

unsigned configured_worker_count() {
    if (const char* s = std::getenv("SWAPTUBE_THREADS")) {
        const int n = std::atoi(s);
        if (n > 0) return static_cast<unsigned>(n);
    }
    const unsigned n = std::thread::hardware_concurrency();
    return n ? n : 4u;
}

// Reusable sense-reversing barrier, shared by the threads of one block.
class Barrier {
public:
    explicit Barrier(unsigned n) : total_(n), remaining_(n) {}

    void arrive() {
        std::unique_lock<std::mutex> lock(mutex_);
        const unsigned my_phase = phase_;
        if (--remaining_ == 0) {
            remaining_ = total_;
            ++phase_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this, my_phase] { return phase_ != my_phase; });
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    const unsigned total_;
    unsigned remaining_;
    unsigned phase_ = 0;
};

// Set only while a barrier-path kernel is running. Null everywhere else, which
// makes __syncthreads() a no-op on the fast path.
thread_local Barrier* tl_barrier = nullptr;

// Fork-join pool. Workers park between launches, so a launch costs one
// broadcast instead of N thread creations -- a 1080p frame issues dozens.
class Pool {
public:
    static Pool& instance() { static Pool p; return p; }

    unsigned size() const { return count_; }

    void run(unsigned num_chunks, const std::function<void(unsigned)>& fn) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = &fn;
            next_chunk_.store(0, std::memory_order_relaxed);
            chunk_total_ = num_chunks;
            outstanding_ = static_cast<unsigned>(workers_.size());
            ++generation_;
        }
        start_.notify_all();

        drain(fn); // the calling thread participates rather than idling

        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return outstanding_ == 0; });
        job_ = nullptr;
    }

private:
    Pool() : count_(configured_worker_count()) {
        // count_ counts the calling thread, so spawn one fewer.
        for (unsigned i = 1; i < count_; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            quit_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    void drain(const std::function<void(unsigned)>& fn) {
        for (;;) {
            const unsigned c = next_chunk_.fetch_add(1, std::memory_order_relaxed);
            if (c >= chunk_total_) return;
            fn(c);
        }
    }

    void worker_loop() {
        unsigned seen = 0;
        for (;;) {
            const std::function<void(unsigned)>* fn = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_.wait(lock, [&] { return quit_ || generation_ != seen; });
                if (quit_) return;
                seen = generation_;
                fn = job_;
            }
            if (fn) drain(*fn);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--outstanding_ == 0) done_.notify_all();
            }
        }
    }

    const unsigned count_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    const std::function<void(unsigned)>* job_ = nullptr;
    std::atomic<unsigned> next_chunk_{0};
    unsigned chunk_total_ = 0;
    unsigned outstanding_ = 0;
    unsigned generation_ = 0;
    bool quit_ = false;
};

} // namespace

void block_barrier() {
    if (tl_barrier) tl_barrier->arrive();
}

unsigned worker_count() { return Pool::instance().size(); }

void run_blocks(unsigned num_blocks, const std::function<void(unsigned, unsigned)>& chunk) {
    const unsigned workers = Pool::instance().size();
    if (workers <= 1 || num_blocks <= 1) { chunk(0, num_blocks); return; }

    // Enough chunks that fast workers can absorb slack on uneven kernels (a
    // fractal interior costs far more than its exterior), but coarse enough
    // that the shared counter stays off the critical path.
    const unsigned target = workers * 8;
    const unsigned per = std::max(1u, (num_blocks + target - 1) / target);
    const unsigned chunks = (num_blocks + per - 1) / per;

    Pool::instance().run(chunks, [&](unsigned c) {
        const unsigned begin = c * per;
        chunk(begin, std::min(begin + per, num_blocks));
    });
}

void run_block_threads(unsigned num_blocks, const dim3& grid, const dim3& block,
                       const std::function<void()>& kernel_body) {
    const unsigned tpb = block.x * block.y * block.z;

    if (tpb == 1) { // nobody to synchronize with
        tl_gridDim = grid;
        tl_blockDim = block;
        tl_threadIdx = dim3(0, 0, 0);
        for (unsigned bi = 0; bi < num_blocks; ++bi) {
            tl_blockIdx = unflatten(bi, grid);
            kernel_body();
        }
        return;
    }

    Barrier barrier(tpb);
    std::vector<std::thread> threads;
    threads.reserve(tpb);

    for (unsigned t = 0; t < tpb; ++t) {
        threads.emplace_back([&, t] {
            tl_gridDim = grid;
            tl_blockDim = block;
            tl_threadIdx = unflatten(t, block);
            tl_barrier = &barrier;

            for (unsigned bi = 0; bi < num_blocks; ++bi) {
                tl_blockIdx = unflatten(bi, grid);
                kernel_body();
                // One block in flight at a time, so __shared__ (a plain
                // static) is not clobbered by the block that follows.
                barrier.arrive();
            }

            tl_barrier = nullptr;
        });
    }
    for (auto& th : threads) th.join();
}

} // namespace swaptube_cpu
