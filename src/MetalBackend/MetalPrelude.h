// Device-side compatibility layer for src/Metal/*.metal.
//
// The shaders reuse src/Host_Device_Shared and src/CUDA/*.cuh verbatim, so this
// header has to supply what nvcc provided implicitly: the f-suffixed math names,
// the bit-cast intrinsics, a stand-in for thrust::complex, and the pixel
// accessors.
//
// <metal_stdlib> and `using namespace metal;` are NOT included here. The runtime
// prepends them when it assembles the source (see MetalRuntime.mm), because the
// build-time include flattener cannot resolve a system Metal header.
#pragma once

// ------------------------------------------------------------------ pixels
// Pixel buffers are typed atomic_uint throughout, because the same buffer is
// written plainly by most kernels and compare-exchanged by the overlay helpers,
// and MSL will not let a device uint* and a device atomic_uint* alias. Relaxed
// atomic load/store compiles to an ordinary load/store on Apple GPUs, so the
// uniform typing costs nothing.
typedef device atomic_uint* PixelBuffer;
typedef const device atomic_uint* ConstPixelBuffer;

inline uint px_load(ConstPixelBuffer pixels, int index) {
    return atomic_load_explicit(&pixels[index], memory_order_relaxed);
}

inline void px_store(PixelBuffer pixels, int index, uint value) {
    atomic_store_explicit(&pixels[index], value, memory_order_relaxed);
}

// Mirrors CUDA's atomicCAS, including its single-shot behaviour: the CUDA code
// does not retry on failure, so a contended pixel drops its update there too.
inline uint px_cas(PixelBuffer pixels, int index, uint expected, uint desired) {
    uint observed = expected;
    atomic_compare_exchange_weak_explicit(&pixels[index], &observed, desired,
                                          memory_order_relaxed, memory_order_relaxed);
    return observed;
}

// -------------------------------------------------------------- math names
// CUDA spells the single-precision functions with an f suffix; MSL overloads the
// unsuffixed names on float.
inline float fminf(float a, float b) { return fmin(a, b); }
inline float fmaxf(float a, float b) { return fmax(a, b); }
inline float rsqrtf(float x) { return rsqrt(x); }
inline float sqrtf(float x) { return sqrt(x); }
inline float powf(float a, float b) { return pow(a, b); }
inline float expf(float x) { return exp(x); }
inline float logf(float x) { return log(x); }
inline float log2f(float x) { return log2(x); }
inline float log10f(float x) { return log10(x); }
inline float sinf(float x) { return sin(x); }
inline float cosf(float x) { return cos(x); }
inline float tanf(float x) { return tan(x); }
inline float asinf(float x) { return asin(x); }
inline float acosf(float x) { return acos(x); }
inline float atanf(float x) { return atan(x); }
inline float atan2f(float y, float x) { return atan2(y, x); }
inline float sinhf(float x) { return sinh(x); }
inline float coshf(float x) { return cosh(x); }
inline float tanhf(float x) { return tanh(x); }
inline float fabsf(float x) { return fabs(x); }
inline float floorf(float x) { return floor(x); }
inline float ceilf(float x) { return ceil(x); }
inline float roundf(float x) { return round(x); }
inline float truncf(float x) { return trunc(x); }
inline float fmodf(float a, float b) { return fmod(a, b); }
inline float hypotf(float a, float b) { return sqrt(a * a + b * b); }
inline float copysignf(float a, float b) { return copysign(a, b); }
inline void sincosf(float a, thread float* s, thread float* c) { *s = sin(a); *c = cos(a); }

inline float __saturatef(float x) { return saturate(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __expf(float x) { return exp(x); }
inline float __logf(float x) { return log(x); }
inline float __sinf(float x) { return sin(x); }
inline float __cosf(float x) { return cos(x); }
inline float __powf(float a, float b) { return pow(a, b); }

inline uint __float_as_uint(float f) { return as_type<uint>(f); }
inline float __uint_as_float(uint u) { return as_type<float>(u); }
inline int __float_as_int(float f) { return as_type<int>(f); }
inline float __int_as_float(int i) { return as_type<float>(i); }
inline int __popc(uint x) { return popcount(x); }
inline int __popcll(ulong x) { return popcount(x); }
inline int __clz(uint x) { return clz(x); }

#ifndef M_PI
#define M_PI M_PI_F
#endif

// vec.h reaches for std::isnan. Metal has no standard library, so the one name
// the shared headers actually use is provided here rather than editing them.
namespace std {
template <typename T> bool isnan(T x) { return ::isnan(x); }
template <typename T> bool isinf(T x) { return ::isinf(x); }
} // namespace std

// ------------------------------------------------------------ thrust stand-in
// The GPU code uses thrust for exactly one thing: thrust::complex<float>. Only
// the operations the kernels actually reach are provided.
namespace thrust {

template <typename T>
struct complex {
    T re, im;

    complex() : re(0), im(0) {}
    complex(T r) : re(r), im(0) {}
    complex(T r, T i) : re(r), im(i) {}

    T real() const { return re; }
    T imag() const { return im; }
};

template <typename T> complex<T> operator-(complex<T> a) { return complex<T>(-a.re, -a.im); }
template <typename T> complex<T> operator+(complex<T> a, complex<T> b) { return complex<T>(a.re + b.re, a.im + b.im); }
template <typename T> complex<T> operator-(complex<T> a, complex<T> b) { return complex<T>(a.re - b.re, a.im - b.im); }
template <typename T> complex<T> operator*(complex<T> a, complex<T> b) { return complex<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
template <typename T> complex<T> operator/(complex<T> a, complex<T> b) {
    T d = b.re * b.re + b.im * b.im;
    return complex<T>((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}
template <typename T> complex<T> operator*(complex<T> a, T s) { return complex<T>(a.re * s, a.im * s); }
template <typename T> complex<T> operator*(T s, complex<T> a) { return complex<T>(a.re * s, a.im * s); }
template <typename T> complex<T> operator/(complex<T> a, T s) { return complex<T>(a.re / s, a.im / s); }
template <typename T> complex<T> operator+(complex<T> a, T s) { return complex<T>(a.re + s, a.im); }
template <typename T> complex<T> operator-(complex<T> a, T s) { return complex<T>(a.re - s, a.im); }

template <typename T> T abs(complex<T> a) { return sqrt(a.re * a.re + a.im * a.im); }
template <typename T> T norm(complex<T> a) { return a.re * a.re + a.im * a.im; }
template <typename T> T arg(complex<T> a) { return atan2(a.im, a.re); }
template <typename T> complex<T> conj(complex<T> a) { return complex<T>(a.re, -a.im); }
template <typename T> complex<T> exp(complex<T> a) { T e = ::exp(a.re); return complex<T>(e * ::cos(a.im), e * ::sin(a.im)); }
template <typename T> complex<T> log(complex<T> a) { return complex<T>(::log(abs(a)), arg(a)); }
template <typename T> complex<T> sqrt(complex<T> a) { T m = ::sqrt(abs(a)); T t = arg(a) * T(0.5); return complex<T>(m * ::cos(t), m * ::sin(t)); }
template <typename T> complex<T> pow(complex<T> a, complex<T> b) { return exp(b * log(a)); }
template <typename T> complex<T> polar(T m, T t) { return complex<T>(m * ::cos(t), m * ::sin(t)); }

} // namespace thrust

// ----------------------------------------------------------- cuComplex stand-in
struct cuFloatComplex { float x, y; };
typedef cuFloatComplex cuComplex;

inline cuFloatComplex make_cuFloatComplex(float r, float i) { cuFloatComplex c; c.x = r; c.y = i; return c; }
inline cuComplex make_cuComplex(float r, float i) { return make_cuFloatComplex(r, i); }
inline float cuCrealf(cuFloatComplex c) { return c.x; }
inline float cuCimagf(cuFloatComplex c) { return c.y; }
inline cuFloatComplex cuCaddf(cuFloatComplex a, cuFloatComplex b) { return make_cuFloatComplex(a.x + b.x, a.y + b.y); }
inline cuFloatComplex cuCsubf(cuFloatComplex a, cuFloatComplex b) { return make_cuFloatComplex(a.x - b.x, a.y - b.y); }
inline cuFloatComplex cuCmulf(cuFloatComplex a, cuFloatComplex b) { return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
inline cuFloatComplex cuCdivf(cuFloatComplex a, cuFloatComplex b) {
    float d = b.x * b.x + b.y * b.y;
    return make_cuFloatComplex((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}
inline float cuCabsf(cuFloatComplex c) { return sqrt(c.x * c.x + c.y * c.y); }
inline cuFloatComplex cuConjf(cuFloatComplex c) { return make_cuFloatComplex(c.x, -c.y); }
