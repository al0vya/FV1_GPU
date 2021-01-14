#pragma once

#include "cuda_runtime.h"

#include "BoundaryConditions.h"

__device__ __forceinline__ real h_init_c_property(BoundaryConditions bcs, real z_int, real x_int)
{
	real eta_west = bcs.hl;
	real eta_east = bcs.hr;

	real h = eta_west - z_int;

	return (x_int <= 25) ? ((h < 0) ? bcs.hl : h) : eta_east - z_int;
}