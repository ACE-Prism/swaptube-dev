// Metal port of the kernel in src/CUDA/mandelbulb.cu.
//
// Profiling MandelbulbDemo at 640x360 put runRaymarch at 99.5% of all kernel
// time (434 ms per frame), which is what makes this the first kernel worth
// moving to the GPU.
//
// The device helpers are not reimplemented here: fractal_sdf.cuh and the rest
// are included straight from src/CUDA. cmake/generate_metal_bindings.py flattens
// the includes and adds Metal's `thread` address space to their reference
// parameters, and MetalPrelude.h supplies the __device__ qualifier and the
// f-suffixed math names nvcc provided implicitly.
//
// <metal_stdlib> is deliberately absent; the runtime prepends it.

#include "../MetalBackend/MetalPrelude.h"

// The CUDA sources mark device functions with these; in MSL every function is
// callable from a kernel, so they carry no meaning and expand away.
#define __device__
#define __forceinline__ inline
#define __host__
#define __global__

#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/Color.h"
#include "../Host_Device_Shared/CameraProjection.h"
#include "../CUDA/fractal_sdf.cuh"

#include "generated/kernel_args.h"

using namespace Cuda;

// Program-scope variables must live in the constant address space in MSL.
constant float EPSILON = 1e-4f;

// get_raymarch_vector out of src/CUDA/common_graphics.cuh. Only this one helper
// is needed, and that header's other functions pull in the atomic pixel path,
// so it is reproduced rather than included.
static inline vec3 raymarch_direction(ivec2 pixel, ivec2 wh, float fov, thread const quat& camera_orientation) {
    float scale = 1 / (sqrt(float(wh.x * wh.y)) * fov);
    vec3 rotated;
    rotated.x = (pixel.x - wh.x * 0.5f) * scale;
    rotated.y = -(pixel.y - wh.y * 0.5f) * scale;
    rotated.z = 1.0f;
    return rotate_vector(rotated, camera_orientation);
}

static inline uint getLighting(thread const vec3& pos, thread const vec3& lightPos,
                               thread const vec3& normal, float shadow,
                               float iters, float max_raymarch_iters) {
    float light = fmax(dot(normal, normalize(lightPos - pos)), 0.25f);
    light *= fmax(shadow, 0.25f);
    float glow = fmin(iters * 2 / max_raymarch_iters, 1.0f);
    return OKLABtoRGB(255, fmin(light + shadow + glow, 1.0f), glow * -0.4f, light * -0.4f);
}

static inline float distMap(thread const vec3& pos, int max_sdf_iters, const int sdfID, const float sdflerp) {
    float dist;
    switch (sdfID) {
        case 1:
            dist = sdf::mandelbulb8(pos, max_sdf_iters);
            break;
        default:
            dist = 0;
            break;
    }
    return fmin(dist, 0.5f);
}

static inline vec4 raymarch(thread const vec3& ro, thread const vec3& rd, const float maxDist,
                            const int max_raymarch_iters, const int max_mandelbulb_iters,
                            const int sdfID, const float sdflerp) {
    vec3 r = ro;
    float d = distMap(r, max_mandelbulb_iters, sdfID, sdflerp);
    float t = d;
    for (int i = 0; i < max_raymarch_iters; i++) {
        r = ro + t * rd;
        d = distMap(r, max_mandelbulb_iters, sdfID, sdflerp);
        if (d < EPSILON) return vec4(r.x, r.y, r.z, (float)i);
        t += d;
        if (t >= maxDist) return vec4(r.x, r.y, r.z, -1.0f);
    }
    return vec4(r.x, r.y, r.z, max_raymarch_iters);
}

static inline float marchLight(thread const vec3& pos, thread const vec3& lightPos, const float minStep,
                               const int max_raymarch_iters, const int max_mandelbulb_iters,
                               const int sdfID, const float sdflerp) {
    vec3 r = pos;
    const vec3 rd = normalize(lightPos - pos);
    float lightDist = length(lightPos - pos);
    float d = minStep;
    float t = d;
    float shadow = 1.0f;
    for (int i = 0; i < max_raymarch_iters; i++) {
        r = pos + t * rd;
        d = distMap(r, max_mandelbulb_iters, sdfID, sdflerp);
        if (d < EPSILON) return 0.0f;
        shadow = fmin(shadow, d / t);
        t += d;
        if (t >= lightDist) return fmax(shadow, 0.0f);
    }
    return 0.0f;
}

static inline vec3 getNormal(thread const vec3& pos, const int max_mandelbulb_iters,
                             const int sdfID, const float sdflerp) {
    return normalize(vec3(
        distMap(pos + vec3(EPSILON, 0, 0), max_mandelbulb_iters, sdfID, sdflerp) - distMap(pos - vec3(EPSILON, 0, 0), max_mandelbulb_iters, sdfID, sdflerp),
        distMap(pos + vec3(0, EPSILON, 0), max_mandelbulb_iters, sdfID, sdflerp) - distMap(pos - vec3(0, EPSILON, 0), max_mandelbulb_iters, sdfID, sdflerp),
        distMap(pos + vec3(0, 0, EPSILON), max_mandelbulb_iters, sdfID, sdflerp) - distMap(pos - vec3(0, 0, EPSILON), max_mandelbulb_iters, sdfID, sdflerp)
    ));
}

kernel void runRaymarch(constant RunRaymarchArgs& args [[buffer(0)]],
                        PixelBuffer colors [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const int pixel_x = int(gid.x);
    const int pixel_y = int(gid.y);
    if (pixel_x >= args.wh.x || pixel_y >= args.wh.y) return;

    const quat camera_orientation = args.camera_orientation;
    const vec3 pos = args.pos;
    const vec3 lightPos = args.lightPos;

    vec3 rd = raymarch_direction(ivec2(pixel_x, pixel_y), args.wh, args.fov, camera_orientation);

    const vec4 rayEnd = raymarch(pos, rd, 10.0f, args.max_raymarch_iters,
                                 args.max_mandelbulb_iters, args.sdfID, args.sdflerp);

    const vec3 end_pos = rayEnd;
    float iters = rayEnd.w;

    uint color;
    if (iters >= 0.0f) {
        float shadow = marchLight(end_pos, lightPos, 8.0f * EPSILON, args.max_raymarch_iters / 4.0f,
                                  args.max_mandelbulb_iters, args.sdfID, args.sdflerp);
        color = getLighting(end_pos, lightPos,
                            getNormal(end_pos, args.max_mandelbulb_iters, args.sdfID, args.sdflerp),
                            shadow, iters, args.max_raymarch_iters);
    } else {
        color = 0xff000000;
    }

    px_store(colors, pixel_y * args.wh.x + pixel_x, color);
}
