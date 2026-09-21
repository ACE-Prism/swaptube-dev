// Metal port of the kernel in src/CUDA/volumetric_fractal.cu.
//
// The heaviest kernel in the project by a wide margin: profiling
// VolumetricFractalDemo at 640x360 put volumeRay at 5.35 seconds per frame, 99.9%
// of all kernel time, because it evaluates the mandelbulb iteration at every step
// along every ray rather than only at the surface.
//
// complex_functions.cuh comes straight from src/CUDA; the generator flattens it
// and adds Metal's `thread` address space to its reference parameters.

#include "../MetalBackend/MetalPrelude.h"

#define __device__
#define __forceinline__ inline
#define __host__
#define __global__

#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/Color.h"
#include "../Host_Device_Shared/CameraProjection.h"
#include "../CUDA/complex_functions.cuh"

#include "generated/kernel_args.h"

using namespace Cuda;

// Program-scope variables must live in the constant address space in MSL.
constant float MAXITERS = 50;

// get_raymarch_vector from src/CUDA/common_graphics.cuh, reproduced because that
// header's other helpers pull in the atomic pixel path this kernel does not need.
static inline vec3 raymarch_direction(ivec2 pixel, ivec2 wh, float fov, thread const quat& camera_orientation) {
    float scale = 1 / (sqrt(float(wh.x * wh.y)) * fov);
    vec3 rotated;
    rotated.x = (pixel.x - wh.x * 0.5f) * scale;
    rotated.y = -(pixel.y - wh.y * 0.5f) * scale;
    rotated.z = 1.0f;
    return rotate_vector(rotated, camera_orientation);
}

static inline vec3 argb_to_vec3(uint argb) {
    return vec3(((argb & 0x00ff0000) >> 16) / 255.0f,
                ((argb & 0x0000ff00) >> 8) / 255.0f,
                ((argb & 0x000000ff)) / 255.0f);
}

static inline uint vec3_to_argb(float alpha, thread const vec3& rgb) {
    return ((int)(alpha * 255.0f) << 24) | ((int)(rgb.x * 255.0f) << 16) |
           ((int)(rgb.y * 255.0f) << 8) | ((int)(rgb.z * 255.0f));
}

static inline float iteration_band_normalize(float iters, float power, float bailout_radius_sq,
                                             float focus_n, float focus_spread) {
    float x = pow(bailout_radius_sq, pow(power, -iters - 1));
    return 1 / (x * x - x);
}

static inline uint sumRay(thread const vec3& ro, thread const vec3& rd, float minDist, float maxDist,
                          float divergence, float stepMult, float dropoff, float opacityMult,
                          float minOpacity, float p) {
    vec3 r = ro;
    vec3 color_accum(0, 0, 0);
    float sq_radius = 0;
    float opacity = 0;
    vec3 z(0, 0, 0);
    float iters = 0;
    float weight = 0;

    float step = divergence * minDist * stepMult;

    for (float t = minDist; t <= maxDist; t += step) {
        step = divergence * t * stepMult;
        r = ro + t * rd;

        iters = mandelbulb_iterations(z, p, r, MAXITERS, 256, sq_radius);

        if (iters == MAXITERS || opacity >= 1.0f) {
            opacity = 1.0f;
            break;
        }

        float c = iters / MAXITERS;

        weight = fmax(fmin(iteration_band_normalize(iters, p, 256, 20, 5), 20 / step) * step * opacityMult * 0.002f,
                      minOpacity);

        color_accum += weight * argb_to_vec3(rainbow(c));

        opacity += weight;
    }

    opacity = fmax(fmin(opacity, 1.0f), 0.0f);
    color_accum = clamp(color_accum, vec3(0, 0, 0), vec3(1, 1, 1));

    return vec3_to_argb(1.0f, color_accum);
}

kernel void volumeRay(constant VolumeRayArgs& args [[buffer(0)]],
                      PixelBuffer colors [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const int pixel_x = int(gid.x);
    const int pixel_y = int(gid.y);
    if (pixel_x >= args.wh.x || pixel_y >= args.wh.y) return;

    // Copied out of the args buffer before use: the struct is bound in the
    // `constant` address space and vec.h's operators take `thread` references,
    // which MSL will not convert between implicitly.
    const quat camera_orientation = args.camera_orientation;
    const vec3 pos = args.pos;
    const vec3 min_corner = args.min_corner;
    const vec3 max_corner = args.max_corner;

    vec3 rd = raymarch_direction(ivec2(pixel_x, pixel_y), args.wh, args.fov, camera_orientation);

    vec3 low_intersections = (min_corner - pos) / rd;
    vec3 high_intersections = (max_corner - pos) / rd;

    vec3 close_intersection = vec3(fmin(low_intersections.x, high_intersections.x),
                                   fmin(low_intersections.y, high_intersections.y),
                                   fmin(low_intersections.z, high_intersections.z));
    vec3 far_intersection = vec3(fmax(low_intersections.x, high_intersections.x),
                                 fmax(low_intersections.y, high_intersections.y),
                                 fmax(low_intersections.z, high_intersections.z));

    float close_dist = fmax(fmax(close_intersection.x, close_intersection.y), close_intersection.z);
    float far_dist = fmin(fmin(far_intersection.x, far_intersection.y), far_intersection.z);

    close_dist = fmax(close_dist, args.min_dist);
    far_dist = fmin(far_dist, args.max_dist);

    const uint color = sumRay(pos, rd, close_dist, far_dist,
                              1 / (sqrt(float(args.wh.x * args.wh.y)) * args.fov),
                              0.5f, 0.92f, 1.0f, 0, args.p);

    px_store(colors, pixel_y * args.wh.x + pixel_x, color);
}
