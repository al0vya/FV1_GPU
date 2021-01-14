#include "friction_update.cuh"

__global__ void friction_update
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	real                 dt,
	AssembledSolution    d_assem_sol
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < sim_params.cells + 2)
	{
		if (d_assem_sol.h_BC[x] > solver_params.tol_dry && abs(d_assem_sol.q_BC[x]) > solver_params.tol_dry)
		{
			real u = d_assem_sol.q_BC[x] / d_assem_sol.h_BC[x];

			real Cf = solver_params.g * pow(sim_params.manning, C(2.0)) / pow(d_assem_sol.h_BC[x], C(1.0) / C(3.0));

			real Sf = -Cf * abs(u) * u;

			real D = 1 + 2 * dt * Cf * abs(u) / d_assem_sol.h_BC[x];

			// Update
			d_assem_sol.q_BC[x] += dt * Sf / D;
		}
	}
}