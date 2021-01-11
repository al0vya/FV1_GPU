#pragma once

#include "cuda_utils.cuh"

#include "BarValues.h"

__host__ void malloc_bar_values
(
	BarValues& d_bar_vals,
	int&       count
);