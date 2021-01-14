#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "h_init_overtopping.cuh"
#include "h_init_c_property.cuh"
#include "bed_data_c_property.cuh"

#include "SimulationParameters.h"
#include "BoundaryConditions.h"
#include "NodalValues.h"

__global__ void get_mesh_and_nodal_values
(
	SimulationParameters sim_params, 
	BoundaryConditions   bcs, 
	NodalValues          d_nodal_vals, 
	real                 dx, 
	int                  test_case
);