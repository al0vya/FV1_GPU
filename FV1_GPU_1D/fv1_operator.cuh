#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "SolverParameters.h"
#include "BarValues.h"
#include "Fluxes.h"
#include "AssembledSolution.h"

__global__ void fv1_operator
(
	SimulationParameters sim_params, 
	SolverParameters     solver_params, 
	real                 dx, 
	real                 dt, 
	int*                 dry_cells, 
	BarValues            d_bar_vals, 
	Fluxes               d_fluxes, 
	AssembledSolution    d_assem_sol
);