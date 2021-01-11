#include "free_fluxes.cuh"

__host__ void free_fluxes(Fluxes& d_fluxes)
{
	free_device(d_fluxes.mass    );
	free_device(d_fluxes.momentum);
}