#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "NodalValues.h"
#include "AssembledSolution.h"

__global__ void get_modal_values
(
	SimulationParameters sim_params,
	NodalValues          d_nodal_vals,
	AssembledSolution    d_assem_sol
);