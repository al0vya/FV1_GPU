#pragma once

#include "cuda_utils.cuh"

#include "FaceValues.h"

__host__ void malloc_face_values
(
	FaceValues& d_face_vals,
	int&        count
);