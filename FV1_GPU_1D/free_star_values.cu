#include "free_star_values.cuh"

__host__ void free_star_values(StarValues& d_star_vals)
{
	free_device(d_star_vals.q_east);
	free_device(d_star_vals.q_west);
	free_device(d_star_vals.h_east);
	free_device(d_star_vals.h_west);
	free_device(d_star_vals.z_east);
	free_device(d_star_vals.z_west);
}