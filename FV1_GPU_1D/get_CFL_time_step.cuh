#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "SolverParameters.h"
#include "AssembledSolution.h"

__global__ void get_CFL_time_step
(
	SimulationParameters sim_params, 
	SolverParameters     solver_params, 
	real                 dx, 
	AssembledSolution    d_assem_sol, 
	real*                d_dtCFLblockLevel
);