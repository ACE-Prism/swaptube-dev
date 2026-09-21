// Host wrapper for the Metal port of src/CUDA/mandelbulb.cu.
//
// Owns the real render_raymarch symbol. The CPU translation of the same file is
// built alongside as render_raymarch_cpu (see cmake/translate_cuda_to_cpu.py),
// which is both the fallback when there is no usable GPU and the reference for
// comparing output: SWAPTUBE_DISABLE_METAL=1 routes every ported kernel back to
// it without rebuilding.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../Host_Device_Shared/vec.h"
#include "../MetalBackend/MetalRuntime.h"
#include "generated/kernel_args.h"

// The CPU translation of this same file.
extern "C" void render_raymarch_cpu(
    const ivec2& wh,
    const vec3& pos, const quat& camera, float fov,
    const vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    const int sdfID, const float sdflerp,
    uint32_t* d_colors);

namespace {

bool metal_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("SWAPTUBE_DISABLE_METAL");
        return s && *s && std::string(s) != "0";
    }();
    return off;
}

} // namespace

extern "C" void render_raymarch(
    const ivec2& wh,
    const vec3& pos, const quat& camera, float fov,
    const vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    const int sdfID, const float sdflerp,
    uint32_t* d_colors
) {
    // The pixel buffer only came from a Metal allocation if the runtime was up
    // when the scene allocated it, so that check covers both failure modes.
    if (metal_disabled() || !swaptube_metal::is_device_pointer(d_colors)) {
        render_raymarch_cpu(wh, pos, camera, fov, lightPos, max_raymarch_iters,
                            max_mandelbulb_iters, sdfID, sdflerp, d_colors);
        return;
    }

    RunRaymarchArgs args;
    args.wh = wh;
    args.pos = pos;
    args.camera_orientation = normalize(camera);
    args.fov = fov;
    args.lightPos = lightPos;
    args.max_raymarch_iters = max_raymarch_iters;
    args.max_mandelbulb_iters = max_mandelbulb_iters;
    args.sdfID = sdfID;
    args.sdflerp = sdflerp;

    swaptube_metal::Dispatch dispatch("runRaymarch");
    if (!dispatch.valid()) {
        render_raymarch_cpu(wh, pos, camera, fov, lightPos, max_raymarch_iters,
                            max_mandelbulb_iters, sdfID, sdflerp, d_colors);
        return;
    }

    dispatch.set_args(&args, sizeof(args));
    dispatch.set_buffer(1, d_colors);
    dispatch.run_2d(wh.x, wh.y);
}
