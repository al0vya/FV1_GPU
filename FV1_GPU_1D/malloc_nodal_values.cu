#include "malloc_nodal_values.cuh"

__host__ void malloc_nodal_values
(
	NodalValues& d_nodal_values, 
	int&         count
)
{
	size_t bytes = count * sizeof(real);

	d_nodal_values.h = (real*)malloc_device(bytes);
	d_nodal_values.q = (real*)malloc_device(bytes);
	d_nodal_values.z = (real*)malloc_device(bytes);
	d_nodal_values.x = (real*)malloc_device(bytes);
}