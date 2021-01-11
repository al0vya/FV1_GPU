#include "malloc_bar_values.cuh"

__host__ void malloc_bar_values
(
	BarValues& d_bar_vals,
	int&       count
)
{
	size_t bytes = count * sizeof(real);

	d_bar_vals.h = (real*)malloc_device(bytes);
	d_bar_vals.z = (real*)malloc_device(bytes);
}