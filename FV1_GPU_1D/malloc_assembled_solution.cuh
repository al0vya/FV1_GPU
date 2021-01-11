#pragma once

#include "cuda_runtime.h"

#include "cuda_utils.cuh"
#include "AssembledSolution.h"

__host__ void malloc_assembled_solution
(
	AssembledSolution& d_assembled_solution,
	int&               count
);