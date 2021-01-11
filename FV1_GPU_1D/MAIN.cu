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
#include "free_nodal_values.cuh"
#include "free_assembled_solution.cuh"

// Sim/solver settings
#include "set_boundary_conditions.h"
#include "set_error_threshold_epsilon.h"
#include "set_num_cells.h"
#include "set_simulation_parameters.h"
#include "set_solver_parameters.h"
#include "set_test_case.h"

__device__ real bed_data_c_property(real x_int);

__device__ real h_init_c_property(BoundaryConditions bcs, real xInt, real hInt);

__device__ real h_init_overtopping(BoundaryConditions bcs, real xInt, real hInt);

__global__ void get_mesh_and_nodal_values(SimulationParameters sim_params, BoundaryConditions bcs, real* xInt, real* zInt, real* hInt, real* qInt, real dx, int test_case)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tx < sim_params.cells + 1)
	{
		xInt[tx] = sim_params.xmin + tx * dx;

		switch (test_case)
		{
		case 1:
		case 2:
		case 3:
			zInt[tx] = 0;
			hInt[tx] = h_init_overtopping(bcs, zInt[tx], xInt[tx]);
			break;
		case 4:
		case 5:
			zInt[tx] = bed_data_c_property(xInt[tx]);
			hInt[tx] = h_init_c_property(bcs, zInt[tx], xInt[tx]);
			break;
		case 6:
			zInt[tx] = bed_data_c_property(xInt[tx]);
			hInt[tx] = h_init_overtopping(bcs, zInt[tx], xInt[tx]);
			break;
		default:
			break;
		}

		qInt[tx] = xInt[tx] <= 32.5 ? bcs.ql : bcs.qr;
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
	real etaWest = bcs.hl;
	real etaEast = bcs.hr;

	real h = etaWest - z_int;

	return (x_int <= 25) ? ((h < 0) ? bcs.hl : h) : etaEast - z_int;
}

__device__ real h_init_overtopping(BoundaryConditions bcs, real z_int, real x_int)
{
	real etaWest = bcs.hl;
	real etaEast = bcs.hr;

	real h;

	h = (x_int <= 25) ? etaWest - z_int : (etaEast - z_int < 0) ? bcs.hr : etaEast - z_int;

	return h;
}

__global__ void get_modal_values(SimulationParameters sim_params, real* qInt, real* hInt, real* zInt, AssembledSolution d_assem_sol)
{
	extern __shared__ real qhzLinear[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	real* q = &qhzLinear[0];
	real* h = &qhzLinear[1 * blockDim.x];
	real* z = &qhzLinear[2 * blockDim.x];

	if (x < sim_params.cells + 1)
	{
		q[tx] = qInt[x];
		h[tx] = hInt[x];
		z[tx] = zInt[x];
	}

	__syncthreads();

	if (x > 0 && x < sim_params.cells + 1)
	{
		if (tx == 0)
		{
			d_assem_sol.q_BC[x] = (qInt[x - 1] + q[tx]) / 2;
			d_assem_sol.h_BC[x] = (hInt[x - 1] + h[tx]) / 2;
			d_assem_sol.z_BC[x] = (zInt[x - 1] + z[tx]) / 2;
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

__global__ void initialiseEtaTemp(SimulationParameters sim_params, AssembledSolution d_assem_sol, real* etaTemp)
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

__global__ void get_wet_dry_cells(SimulationParameters sim_params, SolverParameters solver_params, AssembledSolution d_assem_sol, int* dryCells)
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

		dryCells[x] = (hMax <= solver_params.tol_dry);
	}
}

__global__ void get_face_values(SimulationParameters sim_params, AssembledSolution d_assem_sol, real* etaTemp, FaceValues d_faceValues)
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
		d_faceValues.qEast[x] = (tx < blockDim.x - 1) ? q[tx + 1] : d_assem_sol.q_BC[x + 1];
		d_faceValues.hEast[x] = (tx < blockDim.x - 1) ? h[tx + 1] : d_assem_sol.h_BC[x + 1];
		d_faceValues.etaEast[x] = (tx < blockDim.x - 1) ? eta[tx + 1] : etaTemp[x + 1];

		d_faceValues.qWest[x] = q[tx];
		d_faceValues.hWest[x] = h[tx];
		d_faceValues.etaWest[x] = eta[tx];
	}

	__syncthreads();

	if (x == 0)
	{
		d_faceValues.etaWest[0] = eta[1] - h[1] + h[0];
	}

	if (x == sim_params.cells + 1)
	{
		d_faceValues.etaEast[sim_params.cells] = etaTemp[sim_params.cells] - d_assem_sol.h_BC[sim_params.cells] + d_assem_sol.h_BC[sim_params.cells + 1];
	}
}

__global__ void get_positivity_preserving_nodal_values(SimulationParameters sim_params, SolverParameters solver_params, FaceValues d_faceValues, StarValues d_starValues)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < sim_params.cells + 1)
	{
		real uEast = (d_faceValues.hEast[x] <= solver_params.tol_dry) ? 0 : d_faceValues.qEast[x] / d_faceValues.hEast[x];
		real uWest = (d_faceValues.hWest[x] <= solver_params.tol_dry) ? 0 : d_faceValues.qWest[x] / d_faceValues.hWest[x];

		real a = d_faceValues.etaEast[x] - d_faceValues.hEast[x];
		real b = d_faceValues.etaWest[x] - d_faceValues.hWest[x];

		real zStarIntermediate = max(a, b);

		a = d_faceValues.etaEast[x] - zStarIntermediate;
		b = d_faceValues.etaWest[x] - zStarIntermediate;

		d_starValues.h_east[x] = max(C(0.0), a);
		d_starValues.h_west[x] = max(C(0.0), b);

		real deltaEast = max(-a, C(0.0));
		real deltaWest = max(-b, C(0.0));

		d_starValues.q_east[x] = uEast * d_starValues.h_east[x];
		d_starValues.q_west[x] = uWest * d_starValues.h_west[x];

		d_starValues.z_east[x] = zStarIntermediate - deltaEast;
		d_starValues.z_west[x] = zStarIntermediate - deltaWest;
	}
}

__global__ void fluxHLL(SimulationParameters sim_params, SolverParameters solver_params, FaceValues d_faceValues, StarValues d_starValues, Fluxes d_fluxes)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	real uEast, uWest, aL, aR, hStar, uStar, aStar, sL, sR, massFL, massFR, momentumFL, momentumFR;

	if (x < sim_params.cells + 1)
	{
		if (d_starValues.h_west[x] <= solver_params.tol_dry && d_starValues.h_east[x] <= solver_params.tol_dry)
		{
			d_fluxes.mass[x] = 0;
			d_fluxes.momentum[x] = 0;
		}
		else
		{
			uEast = (d_starValues.h_east[x] <= solver_params.tol_dry) ? 0 : d_starValues.q_east[x] / d_starValues.h_east[x];
			uWest = (d_starValues.h_west[x] <= solver_params.tol_dry) ? 0 : d_starValues.q_west[x] / d_starValues.h_west[x];
			
			aL = sqrt(solver_params.g * d_starValues.h_west[x]);
			aR = sqrt(solver_params.g * d_starValues.h_east[x]);

			hStar = pow(((aL + aR) / 2 + (uWest - uEast) / 4), C(2.0)) / solver_params.g;

			uStar = (uWest + uEast) / 2 + aL - aR;

			aStar = sqrt(solver_params.g * hStar);

			sL = (d_starValues.h_west[x] <= solver_params.tol_dry) ? uEast - 2 * aR : min(uWest - aL, uStar - aStar);
			sR = (d_starValues.h_east[x] <= solver_params.tol_dry) ? uWest + 2 * aL : max(uEast + aR, uStar - aStar);

			massFL = d_starValues.q_west[x];
			massFR = d_starValues.q_east[x];

			momentumFL = uWest * d_starValues.q_west[x] + solver_params.g / 2 * pow(d_starValues.h_west[x], C(2.0));
			momentumFR = uEast * d_starValues.q_east[x] + solver_params.g / 2 * pow(d_starValues.h_east[x], C(2.0));

			if (sL >= 0)
			{
				d_fluxes.mass[x] = massFL;
				d_fluxes.momentum[x] = momentumFL;
			}
			else if (sL < 0 && sR >= 0)
			{
				d_fluxes.mass[x] = (sR * massFL - sL * massFR + sL * sR * (d_starValues.h_east[x] - d_starValues.h_west[x])) / (sR - sL);
				d_fluxes.momentum[x] = (sR * momentumFL - sL * momentumFR + sL * sR * (d_starValues.q_east[x] - d_starValues.q_west[x])) / (sR - sL);
			}
			else if (sR < 0)
			{
				d_fluxes.mass[x] = massFR;
				d_fluxes.momentum[x] = momentumFR;
			}
		}
	}
}

__global__ void get_bar_values(SimulationParameters sim_params, StarValues d_starValues, BarValues d_barValues)
{
	extern __shared__ real zShared[];

	int tx = threadIdx.x; 
	int x = blockIdx.x * blockDim.x + tx;
	
	if (x < sim_params.cells)
	{
		d_barValues.h[x] = (d_starValues.h_west[x + 1] + d_starValues.h_east[x]) / 2;

		d_barValues.z[x] = (d_starValues.z_west[x + 1] - d_starValues.z_east[x]) / (2 * sqrt(C(3.0)));		
	}
}

__global__ void fv1_operator(SimulationParameters sim_params, SolverParameters solver_params, real dx, real dt, int* dryCells, BarValues d_barValues, Fluxes d_fluxes, AssembledSolution d_assem_sol)
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
		if (!dryCells[x])
		{
			
			real m = (tx > 0) ? mass[tx - 1] : d_fluxes.mass[x - 1];
			real p = (tx > 0) ? momentum[tx - 1] : d_fluxes.momentum[x - 1];

			real massIncrement = -(1 / dx) * (mass[tx] - m);
			real momentumIncrement = -(1 / dx) * (momentum[tx] - p + 2 * sqrt(C(3.0)) * solver_params.g * d_barValues.h[x - 1] * d_barValues.z[x - 1]);

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
int smemPerArray(int threadsPerBlock);

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

	// Variables
	real dx = (sim_params.xmax - sim_params.xmin) / sim_params.cells;

	int interfaces = sim_params.cells + 1;
	int sizeInterfaces = interfaces * sizeof(real);

	// Memory allocation


	// ============================================================ //

	real* d_xInt;
	real* d_qInt;
	real* d_hInt;
	real* d_zInt;

	cudaMalloc(&d_xInt, sizeInterfaces);
	cudaMalloc(&d_qInt, sizeInterfaces);
	cudaMalloc(&d_hInt, sizeInterfaces);
	cudaMalloc(&d_zInt, sizeInterfaces);
	checkCUDAError("cudaMalloc for interface values failed");

	int threadsPerBlock = 128;
	int numBlocks = (sim_params.cells + 2) / threadsPerBlock + ((sim_params.cells + 2) % threadsPerBlock != 0);

	dim3 gridDims(numBlocks); // 1D grid dimensions are simply the number of blocks
	dim3 blockDims(threadsPerBlock);

	get_mesh_and_nodal_values<<<gridDims, blockDims>>>(sim_params, bcs, d_xInt, d_zInt, d_hInt, d_qInt, dx, test_case);
	checkCUDAError("kernel get_mesh_and_nodal_values failed");

	cudaDeviceSynchronize();

	int sizeIncBCs = (sim_params.cells + 2) * sizeof(real);

	AssembledSolution d_assem_sol;
	
	real* d_etaTemp;

	cudaMalloc(&d_assem_sol.q_BC, sizeIncBCs);
	cudaMalloc(&d_assem_sol.h_BC, sizeIncBCs);
	cudaMalloc(&d_assem_sol.z_BC, sizeIncBCs);
	cudaMalloc(&d_etaTemp, sizeIncBCs);
	checkCUDAError("cudaMalloc for withBC values failed");

	real* checker = (real*)malloc(sizeIncBCs);
	real* checker2 = (real*)malloc(sizeIncBCs);
	real* checker3 = (real*)malloc(sizeIncBCs);
	real* checker4 = (real*)malloc(sizeIncBCs);

	int smemSize = 3 * smemPerArray(threadsPerBlock); // 3 sets of arrays
	get_modal_values<<<gridDims, blockDims, smemSize>>>(sim_params, d_qInt, d_hInt, d_zInt, d_assem_sol);
	cudaDeviceSynchronize();
	checkCUDAError("kernel get_modal_values failed");


	int* d_dryCells;
	cudaMalloc(&d_dryCells, (sim_params.cells + 2) * sizeof(int));
	checkCUDAError("cudaMalloc for dryCells failed");

	int* h_dryCells = (int*)malloc((sim_params.cells + 2) * sizeof(int));

	FaceValues d_faceValues;

	cudaMalloc(&d_faceValues.qEast, sizeInterfaces);
	cudaMalloc(&d_faceValues.hEast, sizeInterfaces);
	cudaMalloc(&d_faceValues.etaEast, sizeInterfaces);
	checkCUDAError("cudaMalloc for east values failed");

	cudaMalloc(&d_faceValues.qWest, sizeInterfaces);
	cudaMalloc(&d_faceValues.hWest, sizeInterfaces);
	cudaMalloc(&d_faceValues.etaWest, sizeInterfaces);
	checkCUDAError("cudaMalloc for west values failed");

	StarValues d_starValues;

	cudaMalloc(&d_starValues.q_east, sizeInterfaces);
	cudaMalloc(&d_starValues.h_east, sizeInterfaces);
	cudaMalloc(&d_starValues.z_east, sizeInterfaces);
	checkCUDAError("cudaMalloc for east star values failed");

	cudaMalloc(&d_starValues.q_west, sizeInterfaces);
	cudaMalloc(&d_starValues.h_west, sizeInterfaces);
	cudaMalloc(&d_starValues.z_west, sizeInterfaces);
	checkCUDAError("cudaMalloc for west star values failed");

	Fluxes d_fluxes;
	cudaMalloc(&d_fluxes.mass, sizeInterfaces);
	cudaMalloc(&d_fluxes.momentum, sizeInterfaces);

	BarValues d_barValues;
	cudaMalloc(&d_barValues.h, sim_params.cells * sizeof(real));
	cudaMalloc(&d_barValues.z, sim_params.cells * sizeof(real));

	real* d_dtCFLblockLevel;
	cudaMalloc(&d_dtCFLblockLevel, numBlocks * sizeof(real));
	real* h_dtCFLblockLevel = (real*)malloc(numBlocks * sizeof(real));

	real timeNow = 0;
	real dt = C(1e-3);

	int steps = 0;

	while (timeNow < sim_params.simulationTime)
	{
		timeNow += dt;

		if (timeNow - sim_params.simulationTime > 0)
		{
			timeNow -= dt;
			dt = sim_params.simulationTime - timeNow;
			timeNow += dt;
		}

		add_ghost_cells<<<gridDims, blockDims>>>(bcs, sim_params, d_assem_sol);
		cudaDeviceSynchronize();
		checkCUDAError("kernel add_ghost_cells failed");


		initialiseEtaTemp<<<gridDims, blockDims>>>(sim_params, d_assem_sol, d_etaTemp);
		cudaDeviceSynchronize();
		checkCUDAError("kernel initialiseEtaTemp failed");


		if (sim_params.manning > 0)
		{
			friction_update<<<gridDims, blockDims>>>(sim_params, solver_params, dt, d_assem_sol);
			cudaDeviceSynchronize();
			checkCUDAError("kernel friction_update failed");

		}

		smemSize = smemPerArray(threadsPerBlock);
		get_wet_dry_cells<<<gridDims, blockDims, smemSize>>>(sim_params, solver_params, d_assem_sol, d_dryCells);
		cudaDeviceSynchronize();
		checkCUDAError("kernel get_wet_dry_cells failed");


		smemSize = 3 * smemPerArray(threadsPerBlock);
		get_face_values<<<gridDims, blockDims, smemSize>>>(sim_params, d_assem_sol, d_etaTemp, d_faceValues);
		cudaDeviceSynchronize();
		checkCUDAError("kernel initialiseInterfaceValues failed");


		get_positivity_preserving_nodal_values<<<gridDims, blockDims, smemSize>>>(sim_params, solver_params, d_faceValues, d_starValues);
		cudaDeviceSynchronize();
		checkCUDAError("kernel get_positivity_preserving_nodal_values failed");


		fluxHLL<<<gridDims, blockDims>>>(sim_params, solver_params, d_faceValues, d_starValues, d_fluxes);
		cudaDeviceSynchronize();
		checkCUDAError("kernel fluxHLL failed");


		smemSize = smemPerArray(threadsPerBlock);
		get_bar_values<<<gridDims, blockDims, smemSize>>>(sim_params, d_starValues, d_barValues);
		cudaDeviceSynchronize();
		checkCUDAError("kernel barValues failed");
		
		smemSize = 2 * smemPerArray(threadsPerBlock);
		fv1_operator<<<gridDims, blockDims, smemSize>>>(sim_params, solver_params, dx, dt, d_dryCells, d_barValues, d_fluxes, d_assem_sol);
		cudaDeviceSynchronize();
		checkCUDAError("kernel fv1_operator failed");


		smemSize = smemPerArray(threadsPerBlock);
		get_CFL_time_step<<<gridDims, blockDims, smemSize>>>(sim_params, solver_params, dx, d_assem_sol, d_dtCFLblockLevel);
		cudaDeviceSynchronize();
		checkCUDAError("kernel get_CFL_time_step failed");

		
		cudaMemcpy(h_dtCFLblockLevel, d_dtCFLblockLevel, numBlocks * sizeof(real), cudaMemcpyDeviceToHost);
		checkCUDAError("failed to copy dtCFL values to host");

		dt = h_dtCFLblockLevel[0];

		for (int i = 1; i < numBlocks; i++)
		{
			dt = min(h_dtCFLblockLevel[i], dt);
		}

		printf("%f s\n", timeNow);
	}

	cudaMemcpy(checker, d_assem_sol.q_BC, sizeIncBCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker2, d_assem_sol.h_BC, sizeIncBCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker3, d_assem_sol.z_BC, sizeIncBCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker4, d_xInt, sizeInterfaces, cudaMemcpyDeviceToHost);

	std::ofstream data;

	data.open("debug.csv");

	data << "x,q,eta,z" << std::endl;

	for (int i = 0; i < sim_params.cells; i++)
	{
		data << (checker4[i] + checker4[i + 1]) / 2 << "," << checker[i + 1] << "," << max(checker2[i + 1] + checker3[i + 1], checker3[i + 1]) << "," << checker3[i + 1] << std::endl;
	}

	data.close();
	
	cudaFree(d_xInt);
	cudaFree(d_qInt);
	cudaFree(d_hInt);
	cudaFree(d_zInt);

	cudaFree(d_assem_sol.q_BC);
	cudaFree(d_assem_sol.h_BC);
	cudaFree(d_assem_sol.z_BC);
	cudaFree(d_etaTemp);

	free(checker);
	free(checker2);
	free(checker3);
	free(checker4);

	free(h_dryCells);
	cudaFree(d_dryCells);

	cudaFree(d_faceValues.qEast);
	cudaFree(d_faceValues.hEast);
	cudaFree(d_faceValues.etaEast);

	cudaFree(d_faceValues.qWest);
	cudaFree(d_faceValues.hWest);
	cudaFree(d_faceValues.etaWest);

	cudaFree(d_starValues.q_east);
	cudaFree(d_starValues.h_east);
	cudaFree(d_starValues.z_east);
	
	cudaFree(d_starValues.q_west);
	cudaFree(d_starValues.h_west);
	cudaFree(d_starValues.z_west);

	cudaFree(d_fluxes.mass);
	cudaFree(d_fluxes.momentum);

	cudaFree(d_barValues.h);
	cudaFree(d_barValues.z);

	cudaFree(d_dtCFLblockLevel);
	free(h_dtCFLblockLevel);

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

int smemPerArray(int threadsPerBlock)
{
	return threadsPerBlock * sizeof(real);
}