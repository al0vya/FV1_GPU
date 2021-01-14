#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "SolverParameters.h"
#include "FaceValues.h"
#include "StarValues.h"

__global__ void get_positivity_preserving_nodal_values
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	FaceValues           d_face_vals,
	StarValues           d_star_vals
);