#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "BoundaryConditions.h"
#include "AssembledSolution.h"

__global__ void add_ghost_cells
(
	BoundaryConditions   bcs,
	SimulationParameters sim_params,
	AssembledSolution    d_assem_sol
);