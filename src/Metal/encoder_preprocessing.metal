// Metal port of the kernel in src/CUDA/encoder_preprocessing.cu.
//
// Not a heavy kernel, but the only one every frame of every project runs: the
// profile put it at 51% of LatexDemo's kernel time and 89% of GeometryDemo's,
// because those scenes are cheap and this pass still touches every pixel.

#include "../MetalBackend/MetalPrelude.h"

#define __device__
#define __forceinline__ inline
#define __host__
#define __global__
#define __restrict__

#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/Color.h"
#include "generated/encoder_preprocessing_device.h"

#include "generated/kernel_args.h"

using namespace Cuda;

// The Y and UV planes belong to an FFmpeg frame rather than to Metal, so the host
// wrapper dispatches into staging buffers and copies out. Plain device pointers:
// every thread owns a distinct 2x2 block, so there is nothing to synchronize.
kernel void argb_to_p010(constant ArgbToP010Args& args [[buffer(0)]],
                         const device uint* argb [[buffer(1)]],
                         device ushort* y_plane [[buffer(2)]],
                         device ushort* uv_plane [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int uv_x = int(gid.x);
    const int uv_y = int(gid.y);

    const int x = uv_x * 2;
    const int y = uv_y * 2;

    if (x >= args.width || y >= args.height) return;

    float u_sum = 0.0f;
    float v_sum = 0.0f;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            const int ix = x + dx;
            const int iy = y + dy;
            if (ix >= args.width || iy >= args.height) continue;

            uint pixel = argb[iy * args.width + ix];
            pixel = color_combine(args.bg, pixel);

            const uchar r = getr(pixel);
            const uchar g = getg(pixel);
            const uchar b = getb(pixel);

            float yf, uf, vf;
            rgb_to_yuv(r, g, b, yf, uf, vf);

            const float y10f = 64.0f + (yf / 255.0f * 876.0f);
            y_plane[iy * args.y_pitch + ix] = clamp10(y10f) << 6;

            u_sum += uf;
            v_sum += vf;
        }
    }

    u_sum *= 0.25f;
    v_sum *= 0.25f;

    // BT.709 limited range: Cb/Cr 64-960 in 10-bit, left-aligned in 16-bit.
    const float u10f = 512.0f + (u_sum * (896.0f / 255.0f));
    const float v10f = 512.0f + (v_sum * (896.0f / 255.0f));

    const int idx = uv_y * args.uv_pitch + uv_x * 2;
    uv_plane[idx + 0] = clamp10(u10f) << 6;
    uv_plane[idx + 1] = clamp10(v10f) << 6;
}
