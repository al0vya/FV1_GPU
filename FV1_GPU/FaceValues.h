#pragma once

#include "real.h"

typedef struct
{
	real* qEast;
	real* hEast;
	real* etaEast;

	real* qWest;
	real* hWest;
	real* etaWest;

} FaceValues;