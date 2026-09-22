// Metal port of the two kernels in src/CUDA/root_fractal.cu.
//
// One thread per polynomial: each finds the roots by Durand-Kerner and splats a
// gradient circle at every root, so the accumulation buffers are written from all
// over the grid at once. Apple silicon has native float atomics, so those adds
// need no compare-exchange loop.

#include "../MetalBackend/MetalPrelude.h"

#define __device__
#define __forceinline__ inline
#define __host__
#define __global__

#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/Color.h"
#include "../Host_Device_Shared/helpers.h"
#include "../CUDA/find_roots.cuh"

#include "generated/kernel_args.h"

using namespace Cuda;

// generated/root_fractal_device.h is deliberately not included. Everything ahead
// of the first __global__ in that file is complex_pow, which is unused, and
// d_gradient_circle, which is replaced below because its accumulation buffers need
// Metal's atomic types. sigmoid sits after the kernel, so it is not in the prefix
// either and is reproduced here.
static inline float sigmoid(float x) {
    return 3 * x * x - 2 * x * x * x;
}

// d_gradient_circle from the .cu, with the four accumulation buffers typed for
// Metal's atomics. Not taken from the extracted prefix because that prefix sees
// them as plain float pointers.
static inline void gradient_circle(float cx, float cy, float radius,
                                   float red, float green, float blue,
                                   FloatBuffer d_alpha, FloatBuffer d_red,
                                   FloatBuffer d_green, FloatBuffer d_blue,
                                   ivec2 wh, float opa) {
    if (cx < 0 || cx >= wh.x || cy < 0 || cy >= wh.y) return;

    const float radius2 = radius * radius;
    for (int x = cx - radius; x < cx + radius; x++) {
        const float sdx = (x - cx) * (x - cx);
        for (int y = cy - radius; y < cy + radius; y++) {
            const float sdy = (y - cy) * (y - cy);
            const float dist2 = (sdx + sdy) / radius2;
            if (dist2 >= 1.0f) continue;

            const float final_opa = opa / (.025f + 160 * dist2 * dist2);
            if (x >= 0 && x < wh.x && y >= 0 && y < wh.y) {
                const int i = y * wh.x + x;
                accum_add(d_alpha, i, final_opa);
                accum_add(d_red, i, red * final_opa);
                accum_add(d_green, i, green * final_opa);
                accum_add(d_blue, i, blue * final_opa);
            }
        }
    }
}

kernel void root_fractal_kernel(constant RootFractalKernelArgs& args [[buffer(0)]],
                                FloatBuffer d_alpha [[buffer(1)]],
                                FloatBuffer d_red [[buffer(2)]],
                                FloatBuffer d_green [[buffer(3)]],
                                FloatBuffer d_blue [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    const ivec2 wh = args.wh;
    const vec2 lx_ty = args.lx_ty;
    const vec2 rx_by = args.rx_by;

    const uint idx = gid;
    const uint ceil_terms = ceil(args.terms);
    const uint total = 1u << ceil_terms;
    if (idx >= total) return;

    cuFloatComplex coeffs[30];
    float red = 0.0f, green = 0.0f, blue = 0.0f;
    for (uint i = 0; i < ceil_terms; i++) {
        const bool bit = (idx >> i) & 1;
        coeffs[i] = bit ? args.c2 : args.c1;

        if (!bit) continue;
        const int mod = i % 3;
             if (mod == 0) red   += 1 << (7 - i / 3);
        else if (mod == 1) green += 1 << (7 - i / 3);
        else               blue  += 1 << (7 - i / 3);
    }

    red /= 255.0f;
    green /= 255.0f;
    blue /= 255.0f;

    // The leading coefficients may be zero, so the degree has to be found.
    int degree = -1;
    for (int i = int(ceil_terms) - 1; i >= 0; i--) {
        if (coeffs[i].x != 0.0f || coeffs[i].y != 0.0f) { degree = i; break; }
    }
    if (degree < 1) return;

    cuFloatComplex roots[30];
    find_roots(coeffs, degree, roots);

    for (int i = 0; i < degree; i++) {
        const vec2 point(cuCrealf(roots[i]), cuCimagf(roots[i]));
        const vec2 pixel = point_to_pixel_in_screen(point, lx_ty, rx_by, vec2(wh));
        gradient_circle(pixel.x, pixel.y, args.radius, red, green, blue,
                        d_alpha, d_red, d_green, d_blue, wh, args.opacity);
    }
}

kernel void finalize_color_kernel(constant FinalizeColorKernelArgs& args [[buffer(0)]],
                                  PixelBuffer d_pixels [[buffer(1)]],
                                  const device atomic<float>* d_alpha [[buffer(2)]],
                                  const device atomic<float>* d_red [[buffer(3)]],
                                  const device atomic<float>* d_green [[buffer(4)]],
                                  const device atomic<float>* d_blue [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
    const ivec2 wh = args.wh;
    const uint total = uint(wh.x * wh.y);
    if (gid >= total) return;

    const int idx = int(gid);
    const float alpha = accum_load(d_alpha, idx);
    const float inv = 1.0f / (alpha + .000001f);
    const float r_mod_in = accum_load(d_red, idx) * inv;
    const float g_mod_in = accum_load(d_green, idx) * inv;
    const float b_mod_in = accum_load(d_blue, idx) * inv;

    float r_mod = sigmoid(sigmoid(r_mod_in));
    float g_mod = sigmoid(sigmoid(g_mod_in));
    float b_mod = sigmoid(sigmoid(b_mod_in));

    const float brightness = args.brightness * 256.0f;
    r_mod *= 256.0f - brightness;
    g_mod *= 256.0f - brightness;
    b_mod *= 256.0f - brightness;

    // Qualified because `using namespace Cuda` leaves Cuda::clamp and metal::clamp
    // equally visible; the CUDA original qualifies it for the same reason.
    const uint a = Cuda::clamp(alpha * 2.5f, 0.0f, 255.9f);
    const uint r = Cuda::clamp(r_mod + brightness, 0.0f, 255.9f);
    const uint g = Cuda::clamp(g_mod + brightness, 0.0f, 255.9f);
    const uint b = Cuda::clamp(b_mod + brightness, 0.0f, 255.9f);

    px_store(d_pixels, idx, argb(a, r, g, b));
}
