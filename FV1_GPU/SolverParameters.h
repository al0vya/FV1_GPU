#pragma once
#include "real.h"

typedef struct SolverParameters
{
	int cells;
	real CFL;
	real tolDry;
	real g;

} SolverParameters;