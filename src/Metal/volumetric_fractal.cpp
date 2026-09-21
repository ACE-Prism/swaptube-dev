// Host wrapper for the Metal port of src/CUDA/volumetric_fractal.cu.
// See mandelbulb.cpp for how the CPU fallback and SWAPTUBE_DISABLE_METAL work.

#include <cstdint>
#include <cstdlib>
#include <string>

#include "../Host_Device_Shared/vec.h"
#include "../MetalBackend/MetalRuntime.h"
#include "generated/kernel_args.h"

extern "C" void render_volume_cpu(
    const ivec2& wh,
    const vec3& pos, const quat& camera, float fov,
    const vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    const float p,
    uint32_t* colors);

namespace {

bool metal_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("SWAPTUBE_DISABLE_METAL");
        return s && *s && std::string(s) != "0";
    }();
    return off;
}

} // namespace

extern "C" void render_volume(
    const ivec2& wh,
    const vec3& pos, const quat& camera, float fov,
    const vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    const float p,
    uint32_t* colors
) {
    if (metal_disabled() || !swaptube_metal::is_device_pointer(colors)) {
        render_volume_cpu(wh, pos, camera, fov, lightPos, max_raymarch_iters,
                          max_mandelbulb_iters, p, colors);
        return;
    }

    VolumeRayArgs args;
    args.wh = wh;
    args.pos = pos;
    args.camera_orientation = normalize(camera);
    args.fov = fov;
    // The bounding box and distance limits are literals in the CUDA wrapper's
    // launch, not parameters of render_volume.
    args.min_corner = vec3(-2, -2, -2);
    args.max_corner = vec3(2, 2, 2);
    args.min_dist = 0.002f;
    args.max_dist = 8;
    args.lightPos = lightPos;
    args.max_raymarch_iters = max_raymarch_iters;
    args.max_mandelbulb_iters = max_mandelbulb_iters;
    args.p = p;

    swaptube_metal::Dispatch dispatch("volumeRay");
    if (!dispatch.valid()) {
        render_volume_cpu(wh, pos, camera, fov, lightPos, max_raymarch_iters,
                          max_mandelbulb_iters, p, colors);
        return;
    }

    dispatch.set_args(&args, sizeof(args));
    dispatch.set_buffer(1, colors);
    dispatch.run_2d(wh.x, wh.y);
}
