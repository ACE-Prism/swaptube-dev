// Some .cu files include <cuda.h> (the driver API header) for the qualifiers
// and vector types. On the CPU backend it is the same shim.
#pragma once
#include "cuda_runtime.h"
