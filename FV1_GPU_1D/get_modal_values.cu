#include "get_modal_values.cuh"

__global__ void get_modal_values
(
	SimulationParameters sim_params,
	NodalValues          d_nodal_vals,
	AssembledSolution    d_assem_sol
)
{
	extern __shared__ real qhzLinear[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	real* q = &qhzLinear[0];
	real* h = &qhzLinear[1 * blockDim.x];
	real* z = &qhzLinear[2 * blockDim.x];

	if (x < sim_params.cells + 1)
	{
		q[tx] = d_nodal_vals.q[x];
		h[tx] = d_nodal_vals.h[x];
		z[tx] = d_nodal_vals.z[x];
	}

	__syncthreads();

	if (x > 0 && x < sim_params.cells + 1)
	{
		if (tx == 0)
		{
			d_assem_sol.q_BC[x] = (d_nodal_vals.q[x - 1] + q[tx]) / 2;
			d_assem_sol.h_BC[x] = (d_nodal_vals.h[x - 1] + h[tx]) / 2;
			d_assem_sol.z_BC[x] = (d_nodal_vals.z[x - 1] + z[tx]) / 2;
		}
		else
		{
			d_assem_sol.q_BC[x] = (q[tx - 1] + q[tx]) / 2;
			d_assem_sol.h_BC[x] = (h[tx - 1] + h[tx]) / 2;
			d_assem_sol.z_BC[x] = (z[tx - 1] + z[tx]) / 2;
		}
	}
}