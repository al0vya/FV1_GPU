#pragma once

#include "cuda_runtime.h"

#include "CHECK_CUDA_ERROR.cuh"
#include "cuda_utils.cuh"
#include "AssembledSolution.h"

__host__ void free_assembled_solution(AssembledSolution& d_assembled_solution);