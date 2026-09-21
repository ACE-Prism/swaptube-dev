// Host wrapper for src/CUDA/pixels_copy.cu, the one place every device pixel
// buffer is allocated.
//
// Backing those allocations with MTLBuffers is what lets Metal and CPU kernels
// share a frame: the buffers use shared storage, so the pointer handed back is
// valid on both sides and neither a ported nor an unported kernel needs a copy.
// The host/device copy entry points therefore stay plain memcpy and just forward
// to the CPU translation.

#include <cstdint>
#include <cstdlib>

#include "../MetalBackend/MetalRuntime.h"

extern "C" void cuda_copy_pixels_to_device_cpu(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_copy_pixels_to_host_cpu(uint32_t* h_pixels, int size, uint32_t* d_pixels);

extern "C" uint32_t* cuda_alloc_pixels_on_device(int size) {
    const size_t bytes = static_cast<size_t>(size < 0 ? 0 : size) * sizeof(uint32_t);

    if (swaptube_metal::available()) {
        if (void* p = swaptube_metal::allocate(bytes)) return static_cast<uint32_t*>(p);
    }
    // Without a GPU the pointer is ordinary host memory, and is_device_pointer()
    // reporting false is what steers the ported kernels back to the CPU.
    return static_cast<uint32_t*>(std::malloc(bytes ? bytes : 1));
}

extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels) {
    if (!d_pixels) return;
    if (swaptube_metal::is_device_pointer(d_pixels)) {
        swaptube_metal::deallocate(d_pixels);
        return;
    }
    std::free(d_pixels);
}

extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels) {
    cuda_copy_pixels_to_device_cpu(h_pixels, size, d_pixels);
}

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels) {
    cuda_copy_pixels_to_host_cpu(h_pixels, size, d_pixels);
}
