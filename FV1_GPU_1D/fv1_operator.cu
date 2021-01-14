#include "fv1_operator.cuh"

__global__ void fv1_operator
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	real                 dx,
	real                 dt,
	int*                 dry_cells,
	BarValues            d_bar_vals,
	Fluxes               d_fluxes,
	AssembledSolution    d_assem_sol
)
{
	extern __shared__ real massMomentumLinear[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	real* mass = &massMomentumLinear[0];
	real* momentum = &massMomentumLinear[blockDim.x];

	if (x < sim_params.cells + 1)
	{
		mass[tx] = d_fluxes.mass[x];
		momentum[tx] = d_fluxes.momentum[x];
	}

	__syncthreads();

	if (x > 0 && x < sim_params.cells + 1)
	{
		if (!dry_cells[x])
		{

			real m = (tx > 0) ? mass[tx - 1] : d_fluxes.mass[x - 1];
			real p = (tx > 0) ? momentum[tx - 1] : d_fluxes.momentum[x - 1];

			real mass_incr = -(1 / dx) * (mass[tx] - m);
			real momentum_incr = -(1 / dx) * (momentum[tx] - p + 2 * sqrt(C(3.0)) * solver_params.g * d_bar_vals.h[x - 1] * d_bar_vals.z[x - 1]);

			d_assem_sol.h_BC[x] += dt * mass_incr;
			d_assem_sol.q_BC[x] = (d_assem_sol.h_BC[x] <= solver_params.tol_dry) ? 0 : d_assem_sol.q_BC[x] + dt * momentum_incr;
		}
	}
}