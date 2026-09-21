// SWAPTUBE_CPU_DEVICE_TU marks the generated src/CPU/ sources, which are the
// CUDA device translation units compiled as ordinary C++. They need the same
// Cuda:: namespacing nvcc and hipcc get, but there is no device compiler to
// define __CUDACC__ for them.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(SWAPTUBE_CPU_DEVICE_TU)

#define HOST_DEVICE __host__ __device__
#define SHARED_FILE_PREFIX namespace Cuda {
#define SHARED_FILE_SUFFIX }

#else

#define HOST_DEVICE
#define SHARED_FILE_PREFIX
#define SHARED_FILE_SUFFIX

#endif
