#pragma once

#include "real.h"

typedef struct
{
	real* q_BC;
	real* h_BC;
	real* z_BC;
	real* dx_BC;
	real* x;
	int* active_indices;
	int length;

} AssembledSolution;