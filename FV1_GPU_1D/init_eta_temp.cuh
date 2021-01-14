#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "AssembledSolution.h"


__global__ void init_eta_temp
(
	SimulationParameters sim_params, 
	AssembledSolution    d_assem_sol, 
	real*                etaTemp
);