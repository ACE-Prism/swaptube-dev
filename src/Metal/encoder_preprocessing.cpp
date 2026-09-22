// Host wrapper for the Metal port of src/CUDA/encoder_preprocessing.cu.
//
// Unlike the other ports, the destination is not a Metal buffer: the Y and UV
// planes belong to an FFmpeg frame, whose allocator gives no guarantee of the page
// alignment newBufferWithBytesNoCopy would need. So the kernel writes into staging
// buffers that are allocated once and reused, and the result is copied out. At
// 1080p that copy is about 6 MB per frame, against a conversion pass this replaces
// that costs several times as much on the CPU.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../Host_Device_Shared/vec.h"
#include "../MetalBackend/MetalRuntime.h"
#include "generated/kernel_args.h"

extern "C" void preprocess_argb_to_p010_cpu(
    const uint32_t* d_argb, uint16_t* d_y_plane, uint16_t* d_uv_plane,
    int fd, size_t obj_size, int width, int height,
    int y_pitch_bytes, int uv_pitch_bytes,
    unsigned long long y_offset, unsigned long long uv_offset, uint32_t bg);

namespace {

bool metal_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("SWAPTUBE_DISABLE_METAL");
        return s && *s && std::string(s) != "0";
    }();
    return off;
}

// Grown on demand and never shrunk: the frame size is fixed for a render, so this
// allocates on the first frame and then stays put.
class Staging {
public:
    void* get(size_t bytes) {
        if (bytes > size_) {
            if (pointer_) swaptube_metal::deallocate(pointer_);
            pointer_ = swaptube_metal::allocate(bytes);
            size_ = pointer_ ? bytes : 0;
        }
        return pointer_;
    }

private:
    void* pointer_ = nullptr;
    size_t size_ = 0;
};

} // namespace

extern "C" void preprocess_argb_to_p010(
    const uint32_t* d_argb, uint16_t* d_y_plane, uint16_t* d_uv_plane,
    int fd, size_t obj_size, int width, int height,
    int y_pitch_bytes, int uv_pitch_bytes,
    unsigned long long y_offset, unsigned long long uv_offset, uint32_t bg
) {
    const int y_pitch = y_pitch_bytes / static_cast<int>(sizeof(uint16_t));
    const int uv_pitch = uv_pitch_bytes / static_cast<int>(sizeof(uint16_t));

    const size_t y_bytes = static_cast<size_t>(y_pitch_bytes) * height;
    const size_t uv_bytes = static_cast<size_t>(uv_pitch_bytes) * ((height + 1) / 2);

    static Staging y_staging, uv_staging;
    void* y_stage = nullptr;
    void* uv_stage = nullptr;

    if (!metal_disabled() && swaptube_metal::is_device_pointer(d_argb)) {
        y_stage = y_staging.get(y_bytes);
        uv_stage = uv_staging.get(uv_bytes);
    }

    if (!y_stage || !uv_stage) {
        preprocess_argb_to_p010_cpu(d_argb, d_y_plane, d_uv_plane, fd, obj_size, width, height,
                                    y_pitch_bytes, uv_pitch_bytes, y_offset, uv_offset, bg);
        return;
    }

    ArgbToP010Args args;
    args.width = width;
    args.height = height;
    args.y_pitch = y_pitch;
    args.uv_pitch = uv_pitch;
    args.bg = bg;

    swaptube_metal::Dispatch dispatch("argb_to_p010");
    if (!dispatch.valid()) {
        preprocess_argb_to_p010_cpu(d_argb, d_y_plane, d_uv_plane, fd, obj_size, width, height,
                                    y_pitch_bytes, uv_pitch_bytes, y_offset, uv_offset, bg);
        return;
    }

    // The kernel covers one 2x2 block per thread, so the grid is half the frame.
    dispatch.set_args(&args, sizeof(args));
    dispatch.set_buffer(1, d_argb);
    dispatch.set_buffer(2, y_stage);
    dispatch.set_buffer(3, uv_stage);
    dispatch.run_2d((width + 1) / 2, (height + 1) / 2);

    std::memcpy(d_y_plane, y_stage, y_bytes);
    std::memcpy(d_uv_plane, uv_stage, uv_bytes);
}
