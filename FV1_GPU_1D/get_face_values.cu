#include "get_face_values.cuh"

__global__ void get_face_values
(
	SimulationParameters sim_params, 
	AssembledSolution    d_assem_sol, 
	real*                etaTemp, 
	FaceValues           d_face_vals
)
{
	extern __shared__ real qhetaLinear[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;


	real* q = &qhetaLinear[0];
	real* h = &qhetaLinear[1 * blockDim.x];
	real* eta = &qhetaLinear[2 * blockDim.x];

	if (x < sim_params.cells + 2)
	{
		q[tx] = d_assem_sol.q_BC[x];
		h[tx] = d_assem_sol.h_BC[x];
		eta[tx] = etaTemp[x];
	}

	__syncthreads();

	if (x < sim_params.cells + 1)
	{
		d_face_vals.q_east[x] = (tx < blockDim.x - 1) ? q[tx + 1] : d_assem_sol.q_BC[x + 1];
		d_face_vals.h_east[x] = (tx < blockDim.x - 1) ? h[tx + 1] : d_assem_sol.h_BC[x + 1];
		d_face_vals.eta_east[x] = (tx < blockDim.x - 1) ? eta[tx + 1] : etaTemp[x + 1];

		d_face_vals.q_west[x] = q[tx];
		d_face_vals.h_west[x] = h[tx];
		d_face_vals.eta_west[x] = eta[tx];
	}

	__syncthreads();

	if (x == 0) d_face_vals.eta_west[0] = eta[1] - h[1] + h[0];

	if (x == sim_params.cells + 1) d_face_vals.eta_east[sim_params.cells] = etaTemp[sim_params.cells] - d_assem_sol.h_BC[sim_params.cells] + d_assem_sol.h_BC[sim_params.cells + 1];
}