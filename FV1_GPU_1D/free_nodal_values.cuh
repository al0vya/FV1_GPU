#pragma once

#include "cuda_runtime.h"

#include "CHECK_CUDA_ERROR.cuh"
#include "cuda_utils.cuh"
#include "NodalValues.h"

__host__ void free_nodal_values(NodalValues& d_nodal_values);