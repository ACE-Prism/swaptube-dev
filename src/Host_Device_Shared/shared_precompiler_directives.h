#pragma once

// Which compiler is consuming this header decides two things: whether the device
// qualifiers exist, and whether the contents belong in the Cuda namespace.
//
// SWAPTUBE_CPU_DEVICE_TU marks the generated src/CPU/ sources, which are the
// CUDA device translation units compiled as ordinary C++. They need the same
// Cuda:: namespacing nvcc and hipcc get, but there is no device compiler to
// define __CUDACC__ for them.
//
// __METAL_VERSION__ is predefined by the Metal compiler for src/Metal/*.metal.
// Metal has no __host__/__device__ to attach, and it forbids the C++ standard
// library outright, which is why every stdlib include in this directory is
// guarded on that macro. src/MetalBackend/MetalPrelude.h supplies the
// replacements before any of these headers are reached.
#if defined(__METAL_VERSION__)

#define HOST_DEVICE
#define SHARED_FILE_PREFIX namespace Cuda {
#define SHARED_FILE_SUFFIX }

#elif defined(__CUDACC__) || defined(__HIPCC__) || defined(SWAPTUBE_CPU_DEVICE_TU)

#define HOST_DEVICE __host__ __device__
#define SHARED_FILE_PREFIX namespace Cuda {
#define SHARED_FILE_SUFFIX }

#else

#define HOST_DEVICE
#define SHARED_FILE_PREFIX
#define SHARED_FILE_SUFFIX

#endif
