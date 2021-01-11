#include "malloc_star_values.cuh"

__host__ void malloc_star_values
(
	StarValues& d_star_vals,
	int&        count
)
{
	size_t bytes = count * sizeof(real);

	d_star_vals.q_east = (real*)malloc_device(bytes);
	d_star_vals.q_west = (real*)malloc_device(bytes);
	d_star_vals.h_east = (real*)malloc_device(bytes);
	d_star_vals.h_west = (real*)malloc_device(bytes);
	d_star_vals.z_east = (real*)malloc_device(bytes);
	d_star_vals.z_west = (real*)malloc_device(bytes);
}