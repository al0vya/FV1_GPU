#include "free_bar_values.cuh"

__host__ void free_bar_values(BarValues& d_bar_vals)
{
	free_device(d_bar_vals.h);
	free_device(d_bar_vals.z);
}