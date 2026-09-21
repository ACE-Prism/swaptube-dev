// Metal port of the kernel in src/CUDA/fractal_2d.cu.
//
// The device helpers ahead of that file's first __global__ are pulled in through
// generated/fractal_2d_device.h, and the generated unpack macro re-creates the kernel
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
#include "generated/fractal_2d_device.h"
#include "generated/kernel_args.h"

using namespace Cuda;

kernel void go(constant GoArgs& args [[buffer(0)]],
          PixelBuffer colors [[buffer(1)]],
          uint2 gid [[thread_position_in_grid]]) {
    SWAPTUBE_UNPACK_GoArgs(args);

    const int pixel_x = int(gid.x);
    const int pixel_y = int(gid.y);
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    // Scaled so squares are square (coordinates from -1, -1 to 1, 1)
    //float ndc_x = ((pixel_x + 0.5f) / fminf(wh.x, wh.y)) * 2.0f - (wh.x / fminf(wh.x, wh.y));
    //float ndc_y = -(((pixel_y + 0.5f) / fminf(wh.x, wh.y)) * 2.0f - (wh.y / fminf(wh.x, wh.y)));

    float ndc_x = ((pixel_x + 1.0f) / wh.x) - 0.5f;
    float ndc_y = -(((pixel_y + 1.0f) / wh.y) - 0.5f);

    vec2 sub_dimensions(sub_dimensions_x, sub_dimensions_y);

    vec2 tile_size = vec2(wh) / sub_dimensions;

    float macro_aspect = (float) wh.x / wh.y;
    float micro_aspect = (wh.x * sub_dimensions.y) / (wh.y * sub_dimensions.x);

    float scaled_x = ndc_x * sub_dimensions.x;
    float scaled_y = ndc_y * sub_dimensions.y;

    float macro_x = roundf(scaled_x);
    float macro_y = roundf(scaled_y);

    float micro_ndc_x = roundf((scaled_x - macro_x) * 2 * micro_aspect * tile_size.x) / tile_size.x;
    float micro_ndc_y = roundf((scaled_y - macro_y) * 2 * tile_size.y) / tile_size.y;

    float macro_ndc_x = (macro_x) / (sub_dimensions.x * 0.5) * macro_aspect;
    float macro_ndc_y = (macro_y) / (sub_dimensions.y * 0.5);

    float log_real_part_exp, sq_radius = 0;
    float bailout_radius_sq = 256.0 * 256.0;

    int iterations = fractal_iterations(
        zrO, ziO,
        a1rO, a1iO, ac1rO, ac1iO, x1rO, x1iO, 
        a2rO, a2iO, ac2rO, ac2iO, x2rO, x2iO, 
        a3rO, a3iO, ac3rO, ac3iO, x3rO, x3iO, 
        a4rO, a4iO, ac4rO, ac4iO, x4rO, x4iO,
        crO, ciO,
        zrX, ziX,
        a1rX, a1iX, ac1rX, ac1iX, x1rX, x1iX, 
        a2rX, a2iX, ac2rX, ac2iX, x2rX, x2iX, 
        a3rX, a3iX, ac3rX, ac3iX, x3rX, x3iX, 
        a4rX, a4iX, ac4rX, ac4iX, x4rX, x4iX,
        crX, ciX,
        zrY, ziY,
        a1rY, a1iY, ac1rY, ac1iY, x1rY, x1iY, 
        a2rY, a2iY, ac2rY, ac2iY, x2rY, x2iY,
        a3rY, a3iY, ac3rY, ac3iY, x3rY, x3iY,
        a4rY, a4iY, ac4rY, ac4iY, x4rY, x4iY,
        crY, ciY,
        zrx, zix,
        a1rx, a1ix, ac1rx, ac1ix, x1rx, x1ix, 
        a2rx, a2ix, ac2rx, ac2ix, x2rx, x2ix, 
        a3rx, a3ix, ac3rx, ac3ix, x3rx, x3ix, 
        a4rx, a4ix, ac4rx, ac4ix, x4rx, x4ix,
        crx, cix,
        zry, ziy,
        a1ry, a1iy, ac1ry, ac1iy, x1ry, x1iy, 
        a2ry, a2iy, ac2ry, ac2iy, x2ry, x2iy,
        a3ry, a3iy, ac3ry, ac3iy, x3ry, x3iy,
        a4ry, a4iy, ac4ry, ac4iy, x4ry, x4iy,
        cry, ciy,
        param_mode,
        max_iterations, bailout_radius_sq, sq_radius,
        macro_ndc_x, macro_ndc_y,
        micro_ndc_x, micro_ndc_y,
        burning, conj
    );
    
    bool bailed_out = iterations < max_iterations;
    if(iterations == max_iterations){
         px_store(colors, pixel_y * wh.x + pixel_x, 0xff000000);
    }
    else{
        float zr = micro_ndc_x * 2;
        float zi = micro_ndc_y * 2;
        px_store(colors, pixel_y * wh.x + pixel_x, vec3_to_argb(1.0, bezier_gradient(vec3(0.0, 0.0, 1.0), sqrtf(smooth_iterations(iterations, 2.0, sq_radius, bailout_radius_sq) / 50.0))));
    }
}
