#include "get_positivity_preserving_nodal_values.cuh"

__global__ void get_positivity_preserving_nodal_values
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	FaceValues           d_face_vals,
	StarValues           d_star_vals
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < sim_params.cells + 1)
	{
		real u_east = (d_face_vals.h_east[x] <= solver_params.tol_dry) ? 0 : d_face_vals.q_east[x] / d_face_vals.h_east[x];
		real u_west = (d_face_vals.h_west[x] <= solver_params.tol_dry) ? 0 : d_face_vals.q_west[x] / d_face_vals.h_west[x];

		real a = d_face_vals.eta_east[x] - d_face_vals.h_east[x];
		real b = d_face_vals.eta_west[x] - d_face_vals.h_west[x];

		real zStarIntermediate = max(a, b);

		a = d_face_vals.eta_east[x] - zStarIntermediate;
		b = d_face_vals.eta_west[x] - zStarIntermediate;

		d_star_vals.h_east[x] = max(C(0.0), a);
		d_star_vals.h_west[x] = max(C(0.0), b);

		real delta_east = max(-a, C(0.0));
		real delta_west = max(-b, C(0.0));

		d_star_vals.q_east[x] = u_east * d_star_vals.h_east[x];
		d_star_vals.q_west[x] = u_west * d_star_vals.h_west[x];

		d_star_vals.z_east[x] = zStarIntermediate - delta_east;
		d_star_vals.z_west[x] = zStarIntermediate - delta_west;
	}
}