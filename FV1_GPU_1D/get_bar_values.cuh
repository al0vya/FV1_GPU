#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SimulationParameters.h"
#include "StarValues.h"
#include "BarValues.h"

__global__ void get_bar_values
(
	SimulationParameters sim_params,
	StarValues           d_star_vals,
	BarValues            d_bar_vals
);