#pragma once

#include "cuda_utils.cuh"

#include "FaceValues.h"

__host__ void free_face_values(FaceValues& d_face_vals);