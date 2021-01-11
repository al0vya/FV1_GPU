#pragma once

#include "cuda_utils.cuh"

#include "Fluxes.h"

__host__ void malloc_fluxes
(
	Fluxes& d_fluxes, 
	int&    count
);