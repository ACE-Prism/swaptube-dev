// Host wrapper for the Metal port of src/CUDA/mandelbrot.cu.
// See mandelbulb.cpp for how the CPU fallback and SWAPTUBE_DISABLE_METAL work.

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "../Host_Device_Shared/vec.h"
#include "../MetalBackend/MetalRuntime.h"
// The kernel takes cuComplex arguments, so the args struct names that type;
// the shader gets it from MetalPrelude.h instead.
#include "../CPUBackend/cuComplex.h"
#include "generated/kernel_args.h"

extern "C" void mandelbrot_render_cpu(
    const ivec2& wh,
    const vec2& lx_ty,
    const vec2& rx_by,
    const std::complex<float>& seed_z, const std::complex<float>& seed_x, const std::complex<float>& seed_c,
    const vec3& pixel_parameter_multipliers,
    int max_iterations,
    float gradation,
    float phase_shift,
    unsigned int internal_color,
    unsigned int* d_colors);

namespace {

bool metal_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("SWAPTUBE_DISABLE_METAL");
        return s && *s && std::string(s) != "0";
    }();
    return off;
}

} // namespace

extern "C" void mandelbrot_render(
    const ivec2& wh,
    const vec2& lx_ty,
    const vec2& rx_by,
    const std::complex<float>& seed_z, const std::complex<float>& seed_x, const std::complex<float>& seed_c,
    const vec3& pixel_parameter_multipliers,
    int max_iterations,
    float gradation,
    float phase_shift,
    unsigned int internal_color,
    unsigned int* d_colors
) {
    if (metal_disabled() || !swaptube_metal::is_device_pointer(d_colors)) {
        mandelbrot_render_cpu(wh, lx_ty, rx_by, seed_z, seed_x, seed_c,
                              pixel_parameter_multipliers, max_iterations, gradation,
                              phase_shift, internal_color, d_colors);
        return;
    }

    MandelbrotKernelArgs args;
    args.wh = wh;
    args.lx_ty = lx_ty;
    args.rx_by = rx_by;
    // The kernel takes cuComplex, which is layout-identical to two floats; the
    // CUDA wrapper does the same conversion with make_cuComplex.
    args.seed_z = {seed_z.real(), seed_z.imag()};
    args.seed_x = {seed_x.real(), seed_x.imag()};
    args.seed_c = {seed_c.real(), seed_c.imag()};
    args.pixel_parameter_multipliers = pixel_parameter_multipliers;
    args.max_iterations = max_iterations;
    args.gradation = gradation;
    args.phase_shift = phase_shift;
    args.internal_color = internal_color;

    swaptube_metal::Dispatch dispatch("mandelbrot_kernel");
    if (!dispatch.valid()) {
        mandelbrot_render_cpu(wh, lx_ty, rx_by, seed_z, seed_x, seed_c,
                              pixel_parameter_multipliers, max_iterations, gradation,
                              phase_shift, internal_color, d_colors);
        return;
    }

    dispatch.set_args(&args, sizeof(args));
    dispatch.set_buffer(1, d_colors);
    dispatch.run_2d(wh.x, wh.y);
}
