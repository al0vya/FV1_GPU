#include "get_bar_values.cuh"

__global__ void get_bar_values
(
	SimulationParameters sim_params,
	StarValues           d_star_vals,
	BarValues            d_bar_vals
)
{
	extern __shared__ real zShared[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	if (x < sim_params.cells)
	{
		d_bar_vals.h[x] = (d_star_vals.h_west[x + 1] + d_star_vals.h_east[x]) / 2;

		d_bar_vals.z[x] = (d_star_vals.z_west[x + 1] - d_star_vals.z_east[x]) / (2 * sqrt(C(3.0)));
	}
}