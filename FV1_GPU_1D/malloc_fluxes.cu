#include "malloc_fluxes.cuh"

__host__ void malloc_fluxes
(
	Fluxes& d_fluxes, 
	int&    count
)
{
	size_t bytes = count * sizeof(real);

	d_fluxes.mass     = (real*)malloc_device(bytes);
	d_fluxes.momentum = (real*)malloc_device(bytes);
}