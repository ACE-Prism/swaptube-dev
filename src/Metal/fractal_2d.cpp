// Host wrapper for the Metal port of src/CUDA/fractal_2d.cu.
// See mandelbulb.cpp for how the CPU fallback and SWAPTUBE_DISABLE_METAL work.

#include <cstdint>
#include <cstdlib>
#include <string>

#include "../Host_Device_Shared/vec.h"
#include "../MetalBackend/MetalRuntime.h"
#include "generated/kernel_args.h"

extern "C" void fractal_2D_render_cpu(
    const ivec2& wh,
    float o[28], float X[28], float Y[28], float x[28], float y[28],
    const float sub_dimensions_x, const float sub_dimensions_y,
    const char burning, const char conj,
    const int param_mode,
    const int max_iterations,
    unsigned int* colors);

namespace {

bool metal_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("SWAPTUBE_DISABLE_METAL");
        return s && *s && std::string(s) != "0";
    }();
    return off;
}

} // namespace

extern "C" void fractal_2D_render(
    const ivec2& wh,
    float o[28], float X[28], float Y[28], float x[28], float y[28],
    const float sub_dimensions_x, const float sub_dimensions_y,
    const char burning, const char conj,
    const int param_mode,
    const int max_iterations,
    unsigned int* colors
) {
    if (metal_disabled() || !swaptube_metal::is_device_pointer(colors)) {
        fractal_2D_render_cpu(wh, o, X, Y, x, y, sub_dimensions_x, sub_dimensions_y,
                              burning, conj, param_mode, max_iterations, colors);
        return;
    }

    // The CUDA launch spells all 140 of these out one array element at a time.
    // The generated struct keeps the same field order, so the five arrays land
    // contiguously; this asserts that rather than assuming it, because a change to
    // the kernel signature would otherwise scramble them silently.
    static_assert(offsetof(GoArgs, ciy) - offsetof(GoArgs, zrO) == 139 * sizeof(float),
                  "the five 28-float blocks are no longer contiguous in GoArgs");

    GoArgs args;
    args.wh = wh;
    float* fields = &args.zrO; // first of the 28*5 origin/multiplier floats
    for (int i = 0; i < 28; ++i) fields[i] = o[i];
    for (int i = 0; i < 28; ++i) fields[28 + i] = X[i];
    for (int i = 0; i < 28; ++i) fields[56 + i] = Y[i];
    for (int i = 0; i < 28; ++i) fields[84 + i] = x[i];
    for (int i = 0; i < 28; ++i) fields[112 + i] = y[i];
    args.sub_dimensions_x = sub_dimensions_x;
    args.sub_dimensions_y = sub_dimensions_y;
    args.burning = burning;
    args.conj = conj;
    args.param_mode = param_mode;
    args.max_iterations = max_iterations;

    swaptube_metal::Dispatch dispatch("go");
    if (!dispatch.valid()) {
        fractal_2D_render_cpu(wh, o, X, Y, x, y, sub_dimensions_x, sub_dimensions_y,
                              burning, conj, param_mode, max_iterations, colors);
        return;
    }

    dispatch.set_args(&args, sizeof(args));
    dispatch.set_buffer(1, colors);
    dispatch.run_2d(wh.x, wh.y);
}
