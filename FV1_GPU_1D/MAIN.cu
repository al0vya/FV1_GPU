#ifdef __INTELLISENSE__
    #ifndef __CUDACC__
        #define __CUDACC__
    #endif
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

// Aliases
#include "real.h"

// Structures
#include "SimulationParameters.h"
#include "SolverParameters.h"
#include "BoundaryConditions.h"
#include "AssembledSolution.h"
#include "FaceValues.h"
#include "StarValues.h"
#include "Fluxes.h"
#include "BarValues.h"
#include "NodalValues.h"

// Memory (de)allocators
#include "malloc_nodal_values.cuh"
#include "malloc_assembled_solution.cuh"
#include "malloc_bar_values.cuh"
#include "malloc_star_values.cuh"
#include "malloc_fluxes.cuh"
#include "malloc_face_values.cuh"
#include "free_nodal_values.cuh"
#include "free_assembled_solution.cuh"
#include "free_bar_values.cuh"
#include "free_star_values.cuh"
#include "free_fluxes.cuh"
#include "free_face_values.cuh"

// Sim/solver settings
#include "set_boundary_conditions.h"
#include "set_error_threshold_epsilon.h"
#include "set_num_cells.h"
#include "set_simulation_parameters.h"
#include "set_solver_parameters.h"
#include "set_test_case.h"

__device__ real bed_data_c_property(real x_int);

__device__ real h_init_c_property(BoundaryConditions bcs, real x_int, real h_int);

__device__ real h_init_overtopping(BoundaryConditions bcs, real x_int, real h_int);

__global__ void get_mesh_and_nodal_values(SimulationParameters sim_params, BoundaryConditions bcs, NodalValues d_nodal_vals, real dx, int test_case)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tx < sim_params.cells + 1)
	{
		d_nodal_vals.x[tx] = sim_params.xmin + tx * dx;

		switch (test_case)
		{
		case 1:
		case 2:
		case 3:
			d_nodal_vals.z[tx] = 0;
			d_nodal_vals.h[tx] = h_init_overtopping(bcs, d_nodal_vals.z[tx], d_nodal_vals.x[tx]);
			break;
		case 4:
		case 5:
			d_nodal_vals.z[tx] = bed_data_c_property(d_nodal_vals.x[tx]);
			d_nodal_vals.h[tx] = h_init_c_property(bcs, d_nodal_vals.z[tx], d_nodal_vals.x[tx]);
			break;
		case 6:
			d_nodal_vals.z[tx] = bed_data_c_property(d_nodal_vals.x[tx]);
			d_nodal_vals.h[tx] = h_init_overtopping(bcs, d_nodal_vals.z[tx], d_nodal_vals.x[tx]);
			break;
		default:
			break;
		}

		d_nodal_vals.q[tx] = d_nodal_vals.x[tx] <= 32.5 ? bcs.ql : bcs.qr;
	}
}

__device__ real bed_data_c_property(real x_int)
{
	real a = x_int;
	real b;

	if (a >= 22 && a < 25)
	{
		b = C(0.05) * a - C(1.1);
	}
	else if (a >= 25 && a <= 28)
	{
		b = C(-0.05) * a + C(1.4);
	}
	else if (a > 8 && a < 12)
	{
		b = C(0.2) - C(0.05) * pow(a - 10, C(2.0));
	}
	else if (a > 39 && a < 46.5)
	{
		b = C(0.3); // 0.3 is a double literal, dangerous to cast it to a float
	}
	else
	{
		b = 0; // whereas this is safe because you're casting an int literal to a real
	}

	return b * 10;
}

__device__ real h_init_c_property(BoundaryConditions bcs, real z_int, real x_int)
{
	real eta_west = bcs.hl;
	real eta_east = bcs.hr;

	real h = eta_west - z_int;

	return (x_int <= 25) ? ((h < 0) ? bcs.hl : h) : eta_east - z_int;
}

__device__ real h_init_overtopping(BoundaryConditions bcs, real z_int, real x_int)
{
	real eta_west = bcs.hl;
	real eta_east = bcs.hr;

	real h;

	h = (x_int <= 25) ? eta_west - z_int : (eta_east - z_int < 0) ? bcs.hr : eta_east - z_int;

	return h;
}

__global__ void get_modal_values(SimulationParameters sim_params, NodalValues d_nodal_vals, AssembledSolution d_assem_sol)
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

__global__ void add_ghost_cells(BoundaryConditions bcs, SimulationParameters sim_params, AssembledSolution d_assem_sol)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x == 0)
	{
		d_assem_sol.q_BC[x] = bcs.q_imposed_up > 0 ? bcs.q_imposed_up : d_assem_sol.q_BC[x + 1];
		d_assem_sol.h_BC[x] = bcs.h_imposed_up > 0 ? bcs.h_imposed_up : d_assem_sol.h_BC[x + 1];
		d_assem_sol.z_BC[x] = d_assem_sol.z_BC[x + 1];
	}

	if (x == sim_params.cells + 1)
	{
		d_assem_sol.q_BC[x] = bcs.q_imposed_down > 0 ? bcs.q_imposed_down : d_assem_sol.q_BC[x - 1];
		d_assem_sol.h_BC[x] = bcs.h_imposed_down > 0 ? bcs.h_imposed_down : d_assem_sol.h_BC[x - 1];
		d_assem_sol.z_BC[x] = d_assem_sol.z_BC[x - 1];
	}
}

__global__ void init_eta_temp(SimulationParameters sim_params, AssembledSolution d_assem_sol, real* etaTemp)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < sim_params.cells + 1)
	{
		etaTemp[x] = d_assem_sol.h_BC[x] + d_assem_sol.z_BC[x];
	}
}

__global__ void friction_update(SimulationParameters sim_params, SolverParameters solver_params, real dt, AssembledSolution d_assem_sol)
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

__global__ void get_wet_dry_cells(SimulationParameters sim_params, SolverParameters solver_params, AssembledSolution d_assem_sol, int* dry_cells)
{
	extern __shared__ real hShared[];
	
	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;	

	if (x < sim_params.cells + 2)
	{
		hShared[tx] = d_assem_sol.h_BC[x];
	}

	__syncthreads();

	real hMax, hBack, hForward, hLocal;

	if (x > 0 && x < sim_params.cells + 1)
	{
		
		// halo at tx = 0 and tx = blockDim.x - 1 (for blockDim.x = 4, tx = 0, 1, 2, 3)
		hBack = (tx > 0) ? hShared[tx - 1] : d_assem_sol.h_BC[x - 1];
		hLocal = hShared[tx];
		hForward = (tx < blockDim.x - 1) ? hShared[tx + 1] : d_assem_sol.h_BC[x + 1];

		hMax = max(hBack, hLocal);
		hMax = max(hMax, hForward);

		dry_cells[x] = (hMax <= solver_params.tol_dry);
	}
}

__global__ void get_face_values(SimulationParameters sim_params, AssembledSolution d_assem_sol, real* etaTemp, FaceValues d_face_vals)
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

	if (x == 0)
	{
		d_face_vals.eta_west[0] = eta[1] - h[1] + h[0];
	}

	if (x == sim_params.cells + 1)
	{
		d_face_vals.eta_east[sim_params.cells] = etaTemp[sim_params.cells] - d_assem_sol.h_BC[sim_params.cells] + d_assem_sol.h_BC[sim_params.cells + 1];
	}
}

__global__ void get_positivity_preserving_nodal_values(SimulationParameters sim_params, SolverParameters solver_params, FaceValues d_face_vals, StarValues d_star_vals)
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

__global__ void fluxHLL(SimulationParameters sim_params, SolverParameters solver_params, FaceValues d_face_vals, StarValues d_star_vals, Fluxes d_fluxes)
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

__global__ void get_bar_values(SimulationParameters sim_params, StarValues d_star_vals, BarValues d_bar_vals)
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

__global__ void fv1_operator(SimulationParameters sim_params, SolverParameters solver_params, real dx, real dt, int* dry_cells, BarValues d_bar_vals, Fluxes d_fluxes, AssembledSolution d_assem_sol)
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

			real massIncrement = -(1 / dx) * (mass[tx] - m);
			real momentumIncrement = -(1 / dx) * (momentum[tx] - p + 2 * sqrt(C(3.0)) * solver_params.g * d_bar_vals.h[x - 1] * d_bar_vals.z[x - 1]);

			d_assem_sol.h_BC[x] += dt * massIncrement;
			d_assem_sol.q_BC[x] = (d_assem_sol.h_BC[x] <= solver_params.tol_dry) ? 0 : d_assem_sol.q_BC[x] + dt * momentumIncrement;
		}
	}
}

__global__ void get_CFL_time_step(SimulationParameters sim_params, SolverParameters solver_params, real dx, AssembledSolution d_assem_sol, real* d_dtCFLblockLevel)
{
	extern __shared__ real dtCFL[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	dtCFL[tx] = C(1e7);

	__syncthreads();

	// no sync here because each tx is unique and a write, nothing is being read from so no risk of trying to access an uninitialised value

	if (x > 0 && x < sim_params.cells + 1)
	{
		if (d_assem_sol.h_BC[x] >= solver_params.tol_dry)
		{
			real u = d_assem_sol.q_BC[x] / d_assem_sol.h_BC[x];
			dtCFL[tx] = solver_params.CFL * dx / (abs(u) + sqrt(solver_params.g * d_assem_sol.h_BC[x]));
		}
	}

	__syncthreads();

	for (int blockStride = blockDim.x / 2; blockStride > 0; blockStride >>= 1)
	{
		if (tx < blockStride)
		{
			dtCFL[tx] = min(dtCFL[tx], dtCFL[tx + blockStride]);
		}

		__syncthreads(); // same logic as before, sync before next read
	}

	if (tx == 0)
	{
		d_dtCFLblockLevel[blockIdx.x] = dtCFL[0];
	}
}

void checkCUDAError(const char* msg);
int smemPerArray(int threads_per_block);

int main()
{
	int test_case = set_test_case();
	int num_cells = set_num_cells();

	clock_t start = clock();

	// ============================================================ //
	// INITIALISATION OF VARIABLES AND INSTANTIANTION OF STRUCTURES //
	// ============================================================ //

	// Structures
	SimulationParameters sim_params    = set_simulation_parameters(test_case, num_cells);
	SolverParameters     solver_params = set_solver_parameters();
	BoundaryConditions   bcs           = set_boundary_conditions(test_case);

	NodalValues d_nodal_vals;
	AssembledSolution d_assem_sol;
	FaceValues d_face_vals;
	BarValues  d_bar_vals;
	StarValues d_star_vals;
	Fluxes     d_fluxes;

	// Variables
	real dx = (sim_params.xmax - sim_params.xmin) / sim_params.cells;

	int interfaces = sim_params.cells + 1;
	int sizeInterfaces = interfaces * sizeof(real);

	int threads_per_block = 128;
	int num_blocks = (sim_params.cells + 2) / threads_per_block + ((sim_params.cells + 2) % threads_per_block != 0);

	int num_bytes_shared_memory;

	// Bytesizes
	size_t bytes_real_BCs = (sim_params.cells + 2) * sizeof(real);
	size_t bytes_int_BCs  = (sim_params.cells + 2) * sizeof(int);
	size_t bytes_CFL = num_blocks * sizeof(real);

	// Memory allocation
	malloc_nodal_values(d_nodal_vals, interfaces);
	malloc_assembled_solution(d_assem_sol, sim_params.cells);
	malloc_face_values(d_face_vals, interfaces);
	malloc_bar_values(d_bar_vals, sim_params.cells);
	malloc_star_values(d_star_vals, interfaces);
	malloc_fluxes(d_fluxes, interfaces);

	// Arrays
	real* d_eta_temp = (real*)malloc_device(bytes_real_BCs);

	int* d_dry_cells = (int*)malloc_device(bytes_int_BCs);
	int* h_dry_cells = (int*)malloc(bytes_int_BCs);

	real* d_dtCFLblockLevel = (real*)malloc_device(bytes_CFL);
	real* h_dtCFLblockLevel = (real*)malloc(bytes_CFL);

	real* checker  = (real*)malloc(bytes_real_BCs);
	real* checker2 = (real*)malloc(bytes_real_BCs);
	real* checker3 = (real*)malloc(bytes_real_BCs);
	real* checker4 = (real*)malloc(bytes_real_BCs);
	
	// ============================================================ //
	
	get_mesh_and_nodal_values<<<num_blocks, threads_per_block>>>(sim_params, bcs, d_nodal_vals, dx, test_case);
	
	checkCUDAError("kernel get_mesh_and_nodal_values failed");

	num_bytes_shared_memory = 3 * smemPerArray(threads_per_block); // 3 sets of arrays
	get_modal_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>(sim_params, d_nodal_vals, d_assem_sol);
	
	checkCUDAError("kernel get_modal_values failed");

	real timeNow = 0;
	real dt = C(1e-3);

	while (timeNow < sim_params.simulationTime)
	{
		timeNow += dt;

		if (timeNow - sim_params.simulationTime > 0)
		{
			timeNow -= dt;
			dt = sim_params.simulationTime - timeNow;
			timeNow += dt;
		}

		add_ghost_cells<<<num_blocks, threads_per_block>>>(bcs, sim_params, d_assem_sol);
		
		checkCUDAError("kernel add_ghost_cells failed");

		init_eta_temp<<<num_blocks, threads_per_block>>>(sim_params, d_assem_sol, d_eta_temp);
		
		checkCUDAError("kernel init_eta_temp failed");

		if (sim_params.manning > 0)
		{
			friction_update<<<num_blocks, threads_per_block>>>(sim_params, solver_params, dt, d_assem_sol);
			
			checkCUDAError("kernel friction_update failed");

		}

		num_bytes_shared_memory = smemPerArray(threads_per_block);
		get_wet_dry_cells<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>(sim_params, solver_params, d_assem_sol, d_dry_cells);
		
		checkCUDAError("kernel get_wet_dry_cells failed");

		num_bytes_shared_memory = 3 * smemPerArray(threads_per_block);
		get_face_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>(sim_params, d_assem_sol, d_eta_temp, d_face_vals);
		
		checkCUDAError("kernel initialiseInterfaceValues failed");


		get_positivity_preserving_nodal_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>(sim_params, solver_params, d_face_vals, d_star_vals);
		
		checkCUDAError("kernel get_positivity_preserving_nodal_values failed");

		fluxHLL<<<num_blocks, threads_per_block>>>(sim_params, solver_params, d_face_vals, d_star_vals, d_fluxes);
		
		checkCUDAError("kernel fluxHLL failed");

		num_bytes_shared_memory = smemPerArray(threads_per_block);
		get_bar_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>(sim_params, d_star_vals, d_bar_vals);
		
		checkCUDAError("kernel barValues failed");
		
		num_bytes_shared_memory = 2 * smemPerArray(threads_per_block);
		fv1_operator<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>(sim_params, solver_params, dx, dt, d_dry_cells, d_bar_vals, d_fluxes, d_assem_sol);
		
		checkCUDAError("kernel fv1_operator failed");

		num_bytes_shared_memory = smemPerArray(threads_per_block);
		get_CFL_time_step<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>(sim_params, solver_params, dx, d_assem_sol, d_dtCFLblockLevel);
		
		checkCUDAError("kernel get_CFL_time_step failed");

		cudaMemcpy(h_dtCFLblockLevel, d_dtCFLblockLevel, num_blocks * sizeof(real), cudaMemcpyDeviceToHost);
		checkCUDAError("failed to copy dtCFL values to host");

		dt = h_dtCFLblockLevel[0];

		for (int i = 1; i < num_blocks; i++) dt = min(h_dtCFLblockLevel[i], dt);

		printf("%f s\n", timeNow);
	}

	cudaMemcpy(checker, d_assem_sol.q_BC, bytes_real_BCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker2, d_assem_sol.h_BC, bytes_real_BCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker3, d_assem_sol.z_BC, bytes_real_BCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker4, d_nodal_vals.x, sizeInterfaces, cudaMemcpyDeviceToHost);

	std::ofstream data;

	data.open("debug.csv");

	data << "x,q,eta,z" << std::endl;

	for (int i = 0; i < sim_params.cells; i++)
	{
		data << (checker4[i] + checker4[i + 1]) / 2 << "," << checker[i + 1] << "," << max(checker2[i + 1] + checker3[i + 1], checker3[i + 1]) << "," << checker3[i + 1] << std::endl;
	}

	data.close();
	
	// =================== //
	// MEMORY DEALLOCATION //
	// =================== //

	free_nodal_values(d_nodal_vals);
	free_assembled_solution(d_assem_sol);
	free_face_values(d_face_vals);
	free_bar_values(d_bar_vals);
	free_star_values(d_star_vals);
	free_fluxes(d_fluxes);
	
	cudaFree(d_eta_temp);

	free(checker);
	free(checker2);
	free(checker3);
	free(checker4);

	free(h_dry_cells);
	cudaFree(d_dry_cells);

	cudaFree(d_dtCFLblockLevel);
	free(h_dtCFLblockLevel);

	// =================== //

	clock_t end = clock();

	real time = (real)(end - start) / CLOCKS_PER_SEC * C(1000.0);
	printf("Execution time measured using clock(): %f ms\n", time);

	return 0;
}

void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

int smemPerArray(int threads_per_block)
{
	return threads_per_block * sizeof(real);
}