// CPU backend: makes src/CUDA/*.cu compile and run as ordinary C++.
//
// src/CUDA/ stays the single source of truth. CMake copies it to src/CPU/ and
// rewrites kernel launches (see cmake/translate_cuda_to_cpu.py), exactly like
// the HIP path runs hipify. This header supplies everything nvcc would have
// provided implicitly: the __global__/__device__ qualifiers, the launch
// coordinate variables, the runtime API, the device intrinsics and the atomics.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>

// ---------------------------------------------------------------- qualifiers
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
// __restrict__ and __noinline__ are deliberately not redefined: clang supports
// both natively, and libc++ headers use __restrict__ themselves.
#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif
#define __constant__

// Only compute_aabb() in graph_spread.cu declares __shared__, and that kernel
// uses __syncthreads, so it runs one block at a time (see LaunchCPU.h). Plain
// static is therefore the correct sharing scope: shared across the block's
// threads, and no two blocks are ever in flight together.
#define __shared__ static

// ---------------------------------------------------------- launch geometry
struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x_ = 1u, unsigned y_ = 1u, unsigned z_ = 1u) : x(x_), y(y_), z(z_) {}
};

namespace swaptube_cpu {
extern thread_local dim3 tl_threadIdx;
extern thread_local dim3 tl_blockIdx;
extern thread_local dim3 tl_blockDim;
extern thread_local dim3 tl_gridDim;
void block_barrier();
} // namespace swaptube_cpu

#define threadIdx (::swaptube_cpu::tl_threadIdx)
#define blockIdx  (::swaptube_cpu::tl_blockIdx)
#define blockDim  (::swaptube_cpu::tl_blockDim)
#define gridDim   (::swaptube_cpu::tl_gridDim)

inline void __syncthreads() { ::swaptube_cpu::block_barrier(); }
inline void __threadfence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline void __threadfence_block() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

// -------------------------------------------------------------- runtime API
using cudaError_t = int;
enum : cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyHostToHost };

// Apple silicon is unified memory: "device" allocations are just host
// allocations, and every H2D/D2H copy degenerates into a memcpy.
template <class T>
inline cudaError_t cudaMalloc(T** p, size_t bytes) {
    *p = static_cast<T*>(std::malloc(bytes ? bytes : 1));
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (n && dst && src) std::memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* dst, int v, size_t n) {
    if (n && dst) std::memset(dst, v, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, cudaMemcpyKind) {
    for (size_t r = 0; r < height; ++r)
        std::memcpy(static_cast<char*>(dst) + r * dpitch,
                    static_cast<const char*>(src) + r * spitch, width);
    return cudaSuccess;
}
template <class Symbol, class T>
inline cudaError_t cudaMemcpyToSymbol(Symbol& symbol, const T* src, size_t n, size_t offset = 0) {
    std::memcpy(reinterpret_cast<char*>(&symbol) + offset, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "no error (CPU backend)"; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }

// External-memory interop exists only to keep encoder_preprocessing.cu
// compiling. Importing a DMA-BUF is a Linux/VA-API mechanism with no macOS
// analogue; the VideoToolbox path writes into a CVPixelBuffer instead and
// never calls these.
using cudaExternalMemory_t = void*;
struct cudaExternalMemoryHandleDesc {
    int type;
    union { int fd; void* win32; } handle;
    unsigned long long size;
    unsigned int flags;
};
struct cudaExternalMemoryBufferDesc { unsigned long long offset, size; unsigned int flags; };
enum { cudaExternalMemoryHandleTypeOpaqueFd = 1, cudaExternalMemoryHandleTypeOpaqueWin32 = 2 };
inline cudaError_t cudaImportExternalMemory(cudaExternalMemory_t*, const cudaExternalMemoryHandleDesc*) { return 1; }
inline cudaError_t cudaExternalMemoryGetMappedBuffer(void**, cudaExternalMemory_t, const cudaExternalMemoryBufferDesc*) { return 1; }
inline cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t) { return cudaSuccess; }

// ------------------------------------------------------- device intrinsics
inline float __saturatef(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
inline unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
inline int __ffs(int x) { return __builtin_ffs(x); }
inline int __ffsll(long long x) { return __builtin_ffsll(x); }
inline int __clz(int x) { return x ? __builtin_clz(static_cast<unsigned>(x)) : 32; }
inline int __clzll(unsigned long long x) { return x ? __builtin_clzll(x) : 64; }
inline int __mul24(int a, int b) { return a * b; }
inline float __fdividef(float a, float b) { return a / b; }
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
inline float __powf(float a, float b) { return std::pow(a, b); }
inline float __expf(float x) { return std::exp(x); }
inline float __logf(float x) { return std::log(x); }
inline float __log2f(float x) { return std::log2(x); }
inline float __sinf(float x) { return std::sin(x); }
inline float __cosf(float x) { return std::cos(x); }
inline float __tanf(float x) { return std::tan(x); }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
// Apple's math.h already declares __sincosf, so only the plain name is added.
inline void sincosf(float a, float* s, float* c) { *s = std::sin(a); *c = std::cos(a); }

// nvcc exposes a set of mixed-type min/max overloads in device code that
// libc++'s std::min/std::max templates cannot deduce (e.g. min(1.0, aFloat)).
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline float min(float a, int b) { return min(a, static_cast<float>(b)); }
inline float max(float a, int b) { return max(a, static_cast<float>(b)); }
inline float min(int a, float b) { return min(static_cast<float>(a), b); }
inline float max(int a, float b) { return max(static_cast<float>(a), b); }
inline double min(double a, double b) { return a < b ? a : b; }
inline double max(double a, double b) { return a > b ? a : b; }
inline double min(double a, float b) { return min(a, static_cast<double>(b)); }
inline double max(double a, float b) { return max(a, static_cast<double>(b)); }
inline double min(float a, double b) { return min(static_cast<double>(a), b); }
inline double max(float a, double b) { return max(static_cast<double>(a), b); }
inline double min(double a, int b) { return min(a, static_cast<double>(b)); }
inline double max(double a, int b) { return max(a, static_cast<double>(b)); }
inline double min(int a, double b) { return min(static_cast<double>(a), b); }
inline double max(int a, double b) { return max(static_cast<double>(a), b); }

// ---------------------------------------------------------------- atomics
// Blocks run concurrently on the worker pool, so these have to be real atomics.
inline int      atomicAdd(int* a, int v)           { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicAdd(unsigned* a, unsigned v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
inline int      atomicSub(int* a, int v)           { return __atomic_fetch_sub(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicSub(unsigned* a, unsigned v) { return __atomic_fetch_sub(a, v, __ATOMIC_RELAXED); }
inline int      atomicExch(int* a, int v)           { return __atomic_exchange_n(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicExch(unsigned* a, unsigned v) { return __atomic_exchange_n(a, v, __ATOMIC_RELAXED); }
inline int      atomicOr(int* a, int v)            { return __atomic_fetch_or(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicOr(unsigned* a, unsigned v)  { return __atomic_fetch_or(a, v, __ATOMIC_RELAXED); }
inline int      atomicAnd(int* a, int v)           { return __atomic_fetch_and(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicAnd(unsigned* a, unsigned v) { return __atomic_fetch_and(a, v, __ATOMIC_RELAXED); }
inline int      atomicXor(int* a, int v)           { return __atomic_fetch_xor(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicXor(unsigned* a, unsigned v) { return __atomic_fetch_xor(a, v, __ATOMIC_RELAXED); }

inline int atomicCAS(int* a, int expected, int desired) {
    __atomic_compare_exchange_n(a, &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected; // CUDA returns the value that was read, updated in place on failure
}
inline unsigned atomicCAS(unsigned* a, unsigned expected, unsigned desired) {
    __atomic_compare_exchange_n(a, &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}
inline unsigned long long atomicCAS(unsigned long long* a, unsigned long long expected, unsigned long long desired) {
    __atomic_compare_exchange_n(a, &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}

inline int      atomicMax(int* a, int v)           { return __atomic_fetch_max(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicMax(unsigned* a, unsigned v) { return __atomic_fetch_max(a, v, __ATOMIC_RELAXED); }
inline int      atomicMin(int* a, int v)           { return __atomic_fetch_min(a, v, __ATOMIC_RELAXED); }
inline unsigned atomicMin(unsigned* a, unsigned v) { return __atomic_fetch_min(a, v, __ATOMIC_RELAXED); }

// float atomicAdd: CAS loop over the bit pattern, same shape as the
// atomicMax_float helper graph_spread.cu already hand-rolls.
inline float atomicAdd(float* a, float v) {
    unsigned* ai = reinterpret_cast<unsigned*>(a);
    unsigned old = __atomic_load_n(ai, __ATOMIC_RELAXED), assumed;
    do {
        assumed = old;
        float sum = __uint_as_float(assumed) + v;
        unsigned desired = __float_as_uint(sum);
        if (__atomic_compare_exchange_n(ai, &old, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            break;
    } while (assumed != old);
    return __uint_as_float(assumed);
}
