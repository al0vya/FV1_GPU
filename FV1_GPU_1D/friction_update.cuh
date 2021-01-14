#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>

#include "SimulationParameters.h"
#include "SolverParameters.h"
#include "AssembledSolution.h"

__global__ void friction_update
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	real                 dt,
	AssembledSolution    d_assem_sol
);