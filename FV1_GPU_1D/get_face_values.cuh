#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "AssembledSolution.h"
#include "FaceValues.h"

__global__ void get_face_values
(
	SimulationParameters sim_params, 
	AssembledSolution    d_assem_sol, 
	real*                etaTemp, 
	FaceValues           d_face_vals
);