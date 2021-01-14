#pragma once

#include "cuda_runtime.h"

#include "BoundaryConditions.h"

__device__ __forceinline__ real h_init_overtopping(BoundaryConditions bcs, real z_int, real x_int)
{
	real eta_west = bcs.hl;
	real eta_east = bcs.hr;

	real h;

	h = (x_int <= 25) ? eta_west - z_int : (eta_east - z_int < 0) ? bcs.hr : eta_east - z_int;

	return h;
}