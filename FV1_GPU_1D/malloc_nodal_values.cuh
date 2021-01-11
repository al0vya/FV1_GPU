#pragma once

#include "cuda_runtime.h"

#include "cuda_utils.cuh"
#include "NodalValues.h"

__host__ void malloc_nodal_values
(
	NodalValues& d_nodal_values, 
	size_t&      count
);