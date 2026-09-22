// Host wrapper for the Metal port of src/CUDA/root_fractal.cu.
// See mandelbulb.cpp for how the CPU fallback and SWAPTUBE_DISABLE_METAL work.

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>

#include "../Host_Device_Shared/vec.h"
#include "../MetalBackend/MetalRuntime.h"
#include "../CPUBackend/cuComplex.h"
#include "generated/kernel_args.h"

extern "C" void draw_root_fractal_cpu(
    uint32_t* d_pixels, const ivec2& wh,
    const std::complex<float>& c1, const std::complex<float>& c2,
    float terms, const vec2& lx_ty, const vec2& rx_by,
    float radius, float opacity, float brightness);

namespace {

bool metal_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("SWAPTUBE_DISABLE_METAL");
        return s && *s && std::string(s) != "0";
    }();
    return off;
}

// The four accumulation buffers, allocated once and reused. The CUDA wrapper
// cudaMallocs and frees them every call, which also leaves them uninitialized:
// that only works because a fresh large allocation happens to come back zeroed.
// Accumulation genuinely requires zero, so this clears them explicitly.
class Accumulators {
public:
    bool reset(size_t bytes) {
        if (bytes > size_) {
            for (float*& b : buffers_) {
                if (b) swaptube_metal::deallocate(b);
                b = static_cast<float*>(swaptube_metal::allocate(bytes));
                if (!b) { size_ = 0; return false; }
            }
            size_ = bytes;
        }
        for (float* b : buffers_) std::memset(b, 0, bytes);
        return true;
    }

    float* alpha() const { return buffers_[0]; }
    float* red() const { return buffers_[1]; }
    float* green() const { return buffers_[2]; }
    float* blue() const { return buffers_[3]; }

private:
    float* buffers_[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t size_ = 0;
};

} // namespace

extern "C" void draw_root_fractal(
    uint32_t* d_pixels, const ivec2& wh,
    const std::complex<float>& c1, const std::complex<float>& c2,
    float terms, const vec2& lx_ty, const vec2& rx_by,
    float radius, float opacity, float brightness
) {
    static Accumulators accumulators;
    const size_t bytes = static_cast<size_t>(wh.x) * wh.y * sizeof(float);

    if (metal_disabled() || !swaptube_metal::is_device_pointer(d_pixels)
            || !accumulators.reset(bytes)) {
        draw_root_fractal_cpu(d_pixels, wh, c1, c2, terms, lx_ty, rx_by,
                              radius, opacity, brightness);
        return;
    }

    swaptube_metal::Dispatch plot("root_fractal_kernel");
    swaptube_metal::Dispatch finalize("finalize_color_kernel");
    if (!plot.valid() || !finalize.valid()) {
        draw_root_fractal_cpu(d_pixels, wh, c1, c2, terms, lx_ty, rx_by,
                              radius, opacity, brightness);
        return;
    }

    RootFractalKernelArgs plot_args;
    plot_args.wh = wh;
    plot_args.c1 = {c1.real(), c1.imag()};
    plot_args.c2 = {c2.real(), c2.imag()};
    plot_args.terms = terms;
    plot_args.lx_ty = lx_ty;
    plot_args.rx_by = rx_by;
    plot_args.radius = radius;
    plot_args.opacity = opacity;

    // One thread per polynomial: 2^ceil(terms) of them.
    const unsigned total = 1u << static_cast<int>(std::ceil(terms));

    plot.set_args(&plot_args, sizeof(plot_args));
    plot.set_buffer(1, accumulators.alpha());
    plot.set_buffer(2, accumulators.red());
    plot.set_buffer(3, accumulators.green());
    plot.set_buffer(4, accumulators.blue());
    plot.run_1d(total);

    FinalizeColorKernelArgs finalize_args;
    finalize_args.wh = wh;
    finalize_args.brightness = brightness;

    finalize.set_args(&finalize_args, sizeof(finalize_args));
    finalize.set_buffer(1, d_pixels);
    finalize.set_buffer(2, accumulators.alpha());
    finalize.set_buffer(3, accumulators.red());
    finalize.set_buffer(4, accumulators.green());
    finalize.set_buffer(5, accumulators.blue());
    finalize.run_1d(static_cast<unsigned>(wh.x) * wh.y);
}
