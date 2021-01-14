#include "fluxHLL.cuh"

__global__ void fluxHLL
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	FaceValues           d_face_vals,
	StarValues           d_star_vals,
	Fluxes               d_fluxes
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	real u_east, u_west, aL, aR, hStar, uStar, aStar, sL, sR, massFL, massFR, momentumFL, momentumFR;

	if (x < sim_params.cells + 1)
	{
		if (d_star_vals.h_west[x] <= solver_params.tol_dry && d_star_vals.h_east[x] <= solver_params.tol_dry)
		{
			d_fluxes.mass[x] = 0;
			d_fluxes.momentum[x] = 0;
		}
		else
		{
			u_east = (d_star_vals.h_east[x] <= solver_params.tol_dry) ? 0 : d_star_vals.q_east[x] / d_star_vals.h_east[x];
			u_west = (d_star_vals.h_west[x] <= solver_params.tol_dry) ? 0 : d_star_vals.q_west[x] / d_star_vals.h_west[x];

			aL = sqrt(solver_params.g * d_star_vals.h_west[x]);
			aR = sqrt(solver_params.g * d_star_vals.h_east[x]);

			hStar = pow(((aL + aR) / 2 + (u_west - u_east) / 4), C(2.0)) / solver_params.g;

			uStar = (u_west + u_east) / 2 + aL - aR;

			aStar = sqrt(solver_params.g * hStar);

			sL = (d_star_vals.h_west[x] <= solver_params.tol_dry) ? u_east - 2 * aR : min(u_west - aL, uStar - aStar);
			sR = (d_star_vals.h_east[x] <= solver_params.tol_dry) ? u_west + 2 * aL : max(u_east + aR, uStar - aStar);

			massFL = d_star_vals.q_west[x];
			massFR = d_star_vals.q_east[x];

			momentumFL = u_west * d_star_vals.q_west[x] + solver_params.g / 2 * pow(d_star_vals.h_west[x], C(2.0));
			momentumFR = u_east * d_star_vals.q_east[x] + solver_params.g / 2 * pow(d_star_vals.h_east[x], C(2.0));

			if (sL >= 0)
			{
				d_fluxes.mass[x] = massFL;
				d_fluxes.momentum[x] = momentumFL;
			}
			else if (sL < 0 && sR >= 0)
			{
				d_fluxes.mass[x] = (sR * massFL - sL * massFR + sL * sR * (d_star_vals.h_east[x] - d_star_vals.h_west[x])) / (sR - sL);
				d_fluxes.momentum[x] = (sR * momentumFL - sL * momentumFR + sL * sR * (d_star_vals.q_east[x] - d_star_vals.q_west[x])) / (sR - sL);
			}
			else if (sR < 0)
			{
				d_fluxes.mass[x] = massFR;
				d_fluxes.momentum[x] = momentumFR;
			}
		}
	}
}