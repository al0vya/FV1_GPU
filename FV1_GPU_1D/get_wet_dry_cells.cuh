#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "SolverParameters.h"
#include "AssembledSolution.h"

__global__ void get_wet_dry_cells
(
	SimulationParameters sim_params, 
	SolverParameters     solver_params, 
	AssembledSolution    d_assem_sol, 
	int*                 dry_cells
);