#pragma once

#include "cuda_utils.cuh"

#include "StarValues.h"

__host__ void malloc_star_values
(
	StarValues& d_star_vals,
	int&        count
);