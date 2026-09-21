// Metal port of the kernel in src/CUDA/mandelbrot.cu.
//
// The device helpers ahead of that file's first __global__ are pulled in through
// generated/mandelbrot_device.h, and the generated unpack macro re-creates the kernel
// parameters as locals, so the body below is the CUDA body with only three
// changes: the entry signature, the thread indexing, and the pixel writes going
// through the atomic-typed accessor.

#include "../MetalBackend/MetalPrelude.h"

#define __device__
#define __forceinline__ inline
#define __host__
#define __global__

#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/Color.h"
#include "generated/mandelbrot_device.h"
#include "generated/kernel_args.h"

using namespace Cuda;

kernel void mandelbrot_kernel(constant MandelbrotKernelArgs& args [[buffer(0)]],
                         PixelBuffer colors [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    SWAPTUBE_UNPACK_MandelbrotKernelArgs(args);

    const int pixel_x = int(gid.x);
    const int pixel_y = int(gid.y);
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;
    ivec2 pixel(pixel_x, pixel_y);
    px_store(colors, pixel_y * wh.x + pixel_x, 0xffff0000); // Temporary color for debugging

    cuComplex z, x, c; 
    float log_real_part_exp, sq_radius = 0;
    compute_z_x_c(pixel, wh, lx_ty, rx_by, seed_z, seed_x, seed_c, pixel_parameter_multipliers, z, x, c, log_real_part_exp);

    // Check if the exponent 'x' is a positive integer
    bool x_is_real = (cuCimagf(x) == 0) && (cuCrealf(x) > 0) && (cuCrealf(x) == (int)cuCrealf(x));
    int intx = cuCrealf(x);

    int iterations;
    if (x_is_real && (intx == 2 || intx == 3)) {
        iterations = mandelbrot_iterations_2or3(z, intx, c, max_iterations, bailout_radius_sq, sq_radius);
    } else {
        iterations = mandelbrot_iterations(z, x, c, max_iterations, bailout_radius_sq, sq_radius);
    }
    
    bool bailed_out = iterations < max_iterations;

    px_store(colors, pixel_y * wh.x + pixel_x, get_mandelbrot_color(iterations, max_iterations, bailed_out, gradation, sq_radius, log_real_part_exp, phase_shift, internal_color));
}
