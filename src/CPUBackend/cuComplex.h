// CPU stand-in for CUDA's cuComplex.h. complex_functions.cuh and the fractal
// kernels include it; they use thrust::complex for the actual arithmetic, so
// only the type and the basic operations are needed here.
#pragma once
#include <cmath>

struct cuFloatComplex { float x, y; };
struct cuDoubleComplex { double x, y; };
using cuComplex = cuFloatComplex;

inline cuFloatComplex make_cuFloatComplex(float r, float i) { return {r, i}; }
inline cuComplex make_cuComplex(float r, float i) { return {r, i}; }
inline cuDoubleComplex make_cuDoubleComplex(double r, double i) { return {r, i}; }

inline float cuCrealf(cuFloatComplex c) { return c.x; }
inline float cuCimagf(cuFloatComplex c) { return c.y; }
inline cuFloatComplex cuCaddf(cuFloatComplex a, cuFloatComplex b) { return {a.x + b.x, a.y + b.y}; }
inline cuFloatComplex cuCsubf(cuFloatComplex a, cuFloatComplex b) { return {a.x - b.x, a.y - b.y}; }
inline cuFloatComplex cuCmulf(cuFloatComplex a, cuFloatComplex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
inline cuFloatComplex cuCdivf(cuFloatComplex a, cuFloatComplex b) {
    const float d = b.x * b.x + b.y * b.y;
    return {(a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d};
}
inline float cuCabsf(cuFloatComplex c) { return std::sqrt(c.x * c.x + c.y * c.y); }
inline cuFloatComplex cuConjf(cuFloatComplex c) { return {c.x, -c.y}; }

inline double cuCreal(cuDoubleComplex c) { return c.x; }
inline double cuCimag(cuDoubleComplex c) { return c.y; }
inline double cuCabs(cuDoubleComplex c) { return std::sqrt(c.x * c.x + c.y * c.y); }
