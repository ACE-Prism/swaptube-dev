#include "MetalRuntime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Metal/generated/metal_sources.h"

namespace swaptube_metal {
namespace {

// Prepended to the embedded shader source. The build-time include flattener
// (clang -E, see cmake/generate_metal_bindings.py) cannot resolve a system Metal
// header, so <metal_stdlib> is added here instead of in the .metal files.
const char* kSourcePreamble =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n";

class Runtime {
public:
    // Deliberately never destroyed. Device pixel buffers outlive main: the LaTeX
    // SVG cache in src/IO/Latex.cpp is a file-scope map of shared_ptr<DevicePointer>,
    // and freeing those during static destruction calls back in here. A singleton
    // with a destructor would already have taken its mutex down by then, which
    // aborts with "mutex lock failed". The runtime lasts as long as the process, so
    // leaking it is the correct lifetime, not a workaround.
    static Runtime& instance() {
        static Runtime* r = new Runtime();
        return *r;
    }

    bool ok() const { return !functions_.empty(); }
    const char* reason() const { return reason_.c_str(); }
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

    // Compiling a pipeline state is slow, so each kernel is compiled once and
    // reused for every frame.
    id<MTLComputePipelineState> pipeline(const char* name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto cached = pipelines_.find(name);
        if (cached != pipelines_.end()) return cached->second;

        id<MTLComputePipelineState> pso = nil;
        auto found = functions_.find(name);
        if (found == functions_.end()) {
            std::fprintf(stderr, "swaptube_metal: kernel '%s' is in no shader library\n", name);
        } else {
            NSError* err = nil;
            pso = [device_ newComputePipelineStateWithFunction:found->second error:&err];
            if (!pso) {
                std::fprintf(stderr, "swaptube_metal: pipeline for '%s' failed: %s\n",
                             name, [[err localizedDescription] UTF8String]);
            }
        }
        pipelines_[name] = pso;
        return pso;
    }

    void track(void* pointer, id<MTLBuffer> buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffers_[pointer] = buffer;
    }

    id<MTLBuffer> lookup(const void* pointer) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buffers_.find(const_cast<void*>(pointer));
        return it == buffers_.end() ? nil : it->second;
    }

    void forget(void* pointer) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffers_.erase(pointer);
    }

private:
    Runtime() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) { reason_ = "no Metal device"; return; }

        queue_ = [device_ newCommandQueue];
        if (!queue_) { reason_ = "could not create a command queue"; return; }

        MTLCompileOptions* options = [MTLCompileOptions new];
        // Safe rather than fast: nvcc does not enable fast math unless asked, and
        // the SDF raymarchers are sensitive to it. A slightly different transcendental
        // makes a ray terminate an iteration early and that pixel visibly changes,
        // which would show up as noise when diffing against the CPU reference.
        options.mathMode = MTLMathModeSafe;

        int count = 0;
        const EmbeddedShader* shaders = embedded_shaders(count);
        if (count == 0) { reason_ = "no shaders were embedded"; return; }

        for (int i = 0; i < count; ++i) {
            const std::string source = std::string(kSourcePreamble) + shaders[i].source;
            NSError* err = nil;
            id<MTLLibrary> lib =
                [device_ newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options
                                        error:&err];
            if (!lib) {
                std::fprintf(stderr, "swaptube_metal: %s failed to compile:\n%s\n",
                             shaders[i].name, [[err localizedDescription] UTF8String]);
                continue;
            }
            libraries_.push_back(lib);
            for (NSString* fn in [lib functionNames]) {
                functions_[[fn UTF8String]] = [lib newFunctionWithName:fn];
            }
        }

        if (functions_.empty()) reason_ = "every shader failed to compile";
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    std::vector<id<MTLLibrary>> libraries_;
    std::string reason_ = "not initialized";
    std::mutex mutex_;
    std::unordered_map<std::string, id<MTLFunction>> functions_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::unordered_map<void*, id<MTLBuffer>> buffers_;
};

} // namespace

bool available() { return Runtime::instance().ok(); }
const char* unavailable_reason() { return Runtime::instance().reason(); }

void* allocate(size_t bytes) {
    Runtime& rt = Runtime::instance();
    if (!rt.ok()) return nullptr;

    // Shared storage is what makes the pointer usable from both sides. On Apple
    // silicon there is one physical pool, so this is not a compromise.
    id<MTLBuffer> buffer = [rt.device() newBufferWithLength:(bytes ? bytes : 1)
                                                   options:MTLResourceStorageModeShared];
    if (!buffer) return nullptr;

    void* pointer = [buffer contents];
    rt.track(pointer, buffer);
    return pointer;
}

void deallocate(void* pointer) {
    if (!pointer) return;
    Runtime::instance().forget(pointer); // releasing the last reference frees it
}

bool is_device_pointer(const void* pointer) {
    return pointer && Runtime::instance().lookup(pointer) != nil;
}

struct Dispatch::Impl {
    id<MTLComputePipelineState> pso = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    bool args_bound = false;
};

Dispatch::Dispatch(const char* kernel_name) : impl_(new Impl) {
    Runtime& rt = Runtime::instance();
    if (!rt.ok()) return;

    impl_->pso = rt.pipeline(kernel_name);
    if (!impl_->pso) return;

    impl_->command_buffer = [rt.queue() commandBuffer];
    impl_->encoder = [impl_->command_buffer computeCommandEncoder];
    [impl_->encoder setComputePipelineState:impl_->pso];
}

Dispatch::~Dispatch() {
    if (impl_->encoder) [impl_->encoder endEncoding];
    delete impl_;
}

bool Dispatch::valid() const { return impl_->encoder != nil; }

void Dispatch::set_args(const void* args, size_t size) {
    if (!impl_->encoder) return;
    // setBytes is the cheap path for small read-only arguments; Metal's limit is
    // 4 KB, which even the 148-parameter fractal_2d kernel stays under.
    [impl_->encoder setBytes:args length:size atIndex:0];
    impl_->args_bound = true;
}

void Dispatch::set_buffer(unsigned index, const void* device_pointer) {
    if (!impl_->encoder) return;
    id<MTLBuffer> buffer = Runtime::instance().lookup(device_pointer);
    if (!buffer) {
        std::fprintf(stderr, "swaptube_metal: pointer %p at index %u is not a Metal buffer\n",
                     device_pointer, index);
        return;
    }
    [impl_->encoder setBuffer:buffer offset:0 atIndex:index];
}

void Dispatch::run_1d(unsigned count) {
    if (!impl_->encoder || count == 0) return;

    const NSUInteger width = std::min<NSUInteger>([impl_->pso maxTotalThreadsPerThreadgroup], 256);
    [impl_->encoder dispatchThreads:MTLSizeMake(count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [impl_->encoder endEncoding];
    impl_->encoder = nil;
    [impl_->command_buffer commit];
    [impl_->command_buffer waitUntilCompleted];
}

void Dispatch::run_2d(unsigned width, unsigned height) {
    if (!impl_->encoder || width == 0 || height == 0) return;

    // 16x16 matches the block size the CUDA wrappers pick for image kernels.
    const NSUInteger max = [impl_->pso maxTotalThreadsPerThreadgroup];
    NSUInteger tw = 16, th = 16;
    while (tw * th > max) { if (th > 1) th /= 2; else tw /= 2; }

    [impl_->encoder dispatchThreads:MTLSizeMake(width, height, 1)
             threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    [impl_->encoder endEncoding];
    impl_->encoder = nil;
    [impl_->command_buffer commit];
    [impl_->command_buffer waitUntilCompleted];
}

} // namespace swaptube_metal
