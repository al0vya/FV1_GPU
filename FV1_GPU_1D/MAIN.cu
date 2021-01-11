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

// Sim/solver settings
#include "set_boundary_conditions.h"
#include "set_error_threshold_epsilon.h"
#include "set_num_cells.h"
#include "set_simulation_parameters.h"
#include "set_solver_parameters.h"
#include "set_test_case.h"

__device__ __constant__ SimulationParameters d_simulationParameters;
__device__ __constant__ SolverParameters d_solverParameters;
__device__ __constant__ BoundaryConditions d_bcs;
__device__ __constant__ real d_dx;
__device__ __constant__ real d_dt;

__device__ real bedDataConservative(real x_int);

__device__ real hInitialConservative(real xInt, real hInt);

__device__ real hInitialOvertopping(real xInt, real hInt);

__global__ void meshAndInitialConditions(real* xInt, real* zInt, real* hInt, real* qInt, int test_case)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tx < d_simulationParameters.cells + 1)
	{
		xInt[tx] = d_simulationParameters.xmin + tx * d_dx;

		switch (test_case)
		{
		case 1:
		case 2:
		case 3:
			zInt[tx] = 0;
			hInt[tx] = hInitialOvertopping(zInt[tx], xInt[tx]);
			break;
		case 4:
		case 5:
			zInt[tx] = bedDataConservative(xInt[tx]);
			hInt[tx] = hInitialConservative(zInt[tx], xInt[tx]);
			break;
		case 6:
			zInt[tx] = bedDataConservative(xInt[tx]);
			hInt[tx] = hInitialOvertopping(zInt[tx], xInt[tx]);
			break;
		default:
			break;
		}

		qInt[tx] = xInt[tx] <= 32.5 ? d_bcs.ql : d_bcs.qr;
	}
}

__device__ real bedDataConservative(real x_int)
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

__device__ real hInitialConservative(real z_int, real x_int)
{
	real etaWest = d_bcs.hl;
	real etaEast = d_bcs.hr;

	real h = etaWest - z_int;

	return (x_int <= 25) ? ((h < 0) ? d_bcs.hl : h) : etaEast - z_int;
}

__device__ real hInitialOvertopping(real z_int, real x_int)
{
	real etaWest = d_bcs.hl;
	real etaEast = d_bcs.hr;

	real h;

	h = (x_int <= 25) ? etaWest - z_int : (etaEast - z_int < 0) ? d_bcs.hr : etaEast - z_int;

	return h;
}

__global__ void modalProjections(real* qInt, real* hInt, real* zInt, AssembledSolution d_assembledSolution)
{
	extern __shared__ real qhzLinear[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	real* q = &qhzLinear[0];
	real* h = &qhzLinear[1 * blockDim.x];
	real* z = &qhzLinear[2 * blockDim.x];

	if (x < d_simulationParameters.cells + 1)
	{
		q[tx] = qInt[x];
		h[tx] = hInt[x];
		z[tx] = zInt[x];
	}

	__syncthreads();

	if (x > 0 && x < d_simulationParameters.cells + 1)
	{
		if (tx == 0)
		{
			d_assembledSolution.q_BC[x] = (qInt[x - 1] + q[tx]) / 2;
			d_assembledSolution.h_BC[x] = (hInt[x - 1] + h[tx]) / 2;
			d_assembledSolution.z_BC[x] = (zInt[x - 1] + z[tx]) / 2;
		}
		else
		{
			d_assembledSolution.q_BC[x] = (q[tx - 1] + q[tx]) / 2;
			d_assembledSolution.h_BC[x] = (h[tx - 1] + h[tx]) / 2;
			d_assembledSolution.z_BC[x] = (z[tx - 1] + z[tx]) / 2;
		}
	}
}

__global__ void addGhostBCs(AssembledSolution d_assembledSolution)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x == 0)
	{
		d_assembledSolution.q_BC[x] = d_bcs.q_imposed_up > 0 ? d_bcs.q_imposed_up : d_assembledSolution.q_BC[x + 1];
		d_assembledSolution.h_BC[x] = d_bcs.h_imposed_up > 0 ? d_bcs.h_imposed_up : d_assembledSolution.h_BC[x + 1];
		d_assembledSolution.z_BC[x] = d_assembledSolution.z_BC[x + 1];
	}

	if (x == d_simulationParameters.cells + 1)
	{
		d_assembledSolution.q_BC[x] = d_bcs.q_imposed_down > 0 ? d_bcs.q_imposed_down : d_assembledSolution.q_BC[x - 1];
		d_assembledSolution.h_BC[x] = d_bcs.h_imposed_down > 0 ? d_bcs.h_imposed_down : d_assembledSolution.h_BC[x - 1];
		d_assembledSolution.z_BC[x] = d_assembledSolution.z_BC[x - 1];
	}
}

__global__ void initialiseEtaTemp(AssembledSolution d_assembledSolution, real* etaTemp)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < d_simulationParameters.cells + 1)
	{
		etaTemp[x] = d_assembledSolution.h_BC[x] + d_assembledSolution.z_BC[x];
	}
}

__global__ void frictionImplicit(AssembledSolution d_assembledSolution)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < d_simulationParameters.cells + 2)
	{
		if (d_assembledSolution.h_BC[x] > d_solverParameters.tol_dry && abs(d_assembledSolution.q_BC[x]) > d_solverParameters.tol_dry)
		{
			real u = d_assembledSolution.q_BC[x] / d_assembledSolution.h_BC[x];

			real Cf = d_solverParameters.g * pow(d_simulationParameters.manning, C(2.0)) / pow(d_assembledSolution.h_BC[x], C(1.0) / C(3.0));

			real Sf = -Cf * abs(u) * u;

			real D = 1 + 2 * d_dt * Cf * abs(u) / d_assembledSolution.h_BC[x];

			// Update
			d_assembledSolution.q_BC[x] += d_dt * Sf / D;
		}
	}
}

__global__ void wetDryCells(AssembledSolution d_assembledSolution, int* dryCells)
{
	extern __shared__ real hShared[];
	
	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;	

	if (x < d_simulationParameters.cells + 2)
	{
		hShared[tx] = d_assembledSolution.h_BC[x];
	}

	__syncthreads();

	real hMax, hBack, hForward, hLocal;

	if (x > 0 && x < d_simulationParameters.cells + 1)
	{
		
		// halo at tx = 0 and tx = blockDim.x - 1 (for blockDim.x = 4, tx = 0, 1, 2, 3)
		hBack = (tx > 0) ? hShared[tx - 1] : d_assembledSolution.h_BC[x - 1];
		hLocal = hShared[tx];
		hForward = (tx < blockDim.x - 1) ? hShared[tx + 1] : d_assembledSolution.h_BC[x + 1];

		hMax = max(hBack, hLocal);
		hMax = max(hMax, hForward);

		dryCells[x] = (hMax <= d_solverParameters.tol_dry);
	}
}

__global__ void initialiseFaceValues(AssembledSolution d_assembledSolution, real* etaTemp, FaceValues d_faceValues)
{
	extern __shared__ real qhetaLinear[];

	int tx = threadIdx.x; 
	int x = blockIdx.x * blockDim.x + tx;
	

	real* q = &qhetaLinear[0];
	real* h = &qhetaLinear[1 * blockDim.x];
	real* eta = &qhetaLinear[2 * blockDim.x];

	if (x < d_simulationParameters.cells + 2)
	{
		q[tx] = d_assembledSolution.q_BC[x];
		h[tx] = d_assembledSolution.h_BC[x];
		eta[tx] = etaTemp[x];
	}

	__syncthreads();

	if (x < d_simulationParameters.cells + 1)
	{
		d_faceValues.qEast[x] = (tx < blockDim.x - 1) ? q[tx + 1] : d_assembledSolution.q_BC[x + 1];
		d_faceValues.hEast[x] = (tx < blockDim.x - 1) ? h[tx + 1] : d_assembledSolution.h_BC[x + 1];
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

	if (x == d_simulationParameters.cells + 1)
	{
		d_faceValues.etaEast[d_simulationParameters.cells] = etaTemp[d_simulationParameters.cells] - d_assembledSolution.h_BC[d_simulationParameters.cells] + d_assembledSolution.h_BC[d_simulationParameters.cells + 1];
	}
}

__global__ void wettingAndDrying(FaceValues d_faceValues, StarValues d_starValues)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < d_simulationParameters.cells + 1)
	{
		real uEast = (d_faceValues.hEast[x] <= d_solverParameters.tol_dry) ? 0 : d_faceValues.qEast[x] / d_faceValues.hEast[x];
		real uWest = (d_faceValues.hWest[x] <= d_solverParameters.tol_dry) ? 0 : d_faceValues.qWest[x] / d_faceValues.hWest[x];

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

__global__ void fluxHLL(FaceValues d_faceValues, StarValues d_starValues, Fluxes d_fluxes)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	real uEast, uWest, aL, aR, hStar, uStar, aStar, sL, sR, massFL, massFR, momentumFL, momentumFR;

	if (x < d_simulationParameters.cells + 1)
	{
		if (d_starValues.h_west[x] <= d_solverParameters.tol_dry && d_starValues.h_east[x] <= d_solverParameters.tol_dry)
		{
			d_fluxes.mass[x] = 0;
			d_fluxes.momentum[x] = 0;
		}
		else
		{
			uEast = (d_starValues.h_east[x] <= d_solverParameters.tol_dry) ? 0 : d_starValues.q_east[x] / d_starValues.h_east[x];
			uWest = (d_starValues.h_west[x] <= d_solverParameters.tol_dry) ? 0 : d_starValues.q_west[x] / d_starValues.h_west[x];
			
			aL = sqrt(d_solverParameters.g * d_starValues.h_west[x]);
			aR = sqrt(d_solverParameters.g * d_starValues.h_east[x]);

			hStar = pow(((aL + aR) / 2 + (uWest - uEast) / 4), C(2.0)) / d_solverParameters.g;

			uStar = (uWest + uEast) / 2 + aL - aR;

			aStar = sqrt(d_solverParameters.g * hStar);

			sL = (d_starValues.h_west[x] <= d_solverParameters.tol_dry) ? uEast - 2 * aR : min(uWest - aL, uStar - aStar);
			sR = (d_starValues.h_east[x] <= d_solverParameters.tol_dry) ? uWest + 2 * aL : max(uEast + aR, uStar - aStar);

			massFL = d_starValues.q_west[x];
			massFR = d_starValues.q_east[x];

			momentumFL = uWest * d_starValues.q_west[x] + d_solverParameters.g / 2 * pow(d_starValues.h_west[x], C(2.0));
			momentumFR = uEast * d_starValues.q_east[x] + d_solverParameters.g / 2 * pow(d_starValues.h_east[x], C(2.0));

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

__global__ void initialiseBarValues(StarValues d_starValues, BarValues d_barValues)
{
	extern __shared__ real zShared[];

	int tx = threadIdx.x; 
	int x = blockIdx.x * blockDim.x + tx;
	
	if (x < d_simulationParameters.cells)
	{
		d_barValues.h[x] = (d_starValues.h_west[x + 1] + d_starValues.h_east[x]) / 2;

		d_barValues.z[x] = (d_starValues.z_west[x + 1] - d_starValues.z_east[x]) / (2 * sqrt(C(3.0)));		
	}
}

__global__ void fv1Operator(int* dryCells, BarValues d_barValues, Fluxes d_fluxes, AssembledSolution d_assembledSolution)
{
	extern __shared__ real massMomentumLinear[];

	int tx = threadIdx.x; 
	int x = blockIdx.x * blockDim.x + tx;
	
	real* mass = &massMomentumLinear[0];
	real* momentum = &massMomentumLinear[blockDim.x];

	if (x < d_simulationParameters.cells + 1)
	{
		mass[tx] = d_fluxes.mass[x];
		momentum[tx] = d_fluxes.momentum[x];
	}

	__syncthreads();

	if (x > 0 && x < d_simulationParameters.cells + 1)
	{
		if (!dryCells[x])
		{
			
			real m = (tx > 0) ? mass[tx - 1] : d_fluxes.mass[x - 1];
			real p = (tx > 0) ? momentum[tx - 1] : d_fluxes.momentum[x - 1];

			real massIncrement = -(1 / d_dx) * (mass[tx] - m);
			real momentumIncrement = -(1 / d_dx) * (momentum[tx] - p + 2 * sqrt(C(3.0)) * d_solverParameters.g * d_barValues.h[x - 1] * d_barValues.z[x - 1]);

			d_assembledSolution.h_BC[x] += d_dt * massIncrement;
			d_assembledSolution.q_BC[x] = (d_assembledSolution.h_BC[x] <= d_solverParameters.tol_dry) ? 0 : d_assembledSolution.q_BC[x] + d_dt * momentumIncrement;
		}
	}
}

__global__ void timeStepAdjustment(AssembledSolution d_assembledSolution, real* dtCFLblockLevel)
{
	extern __shared__ real dtCFL[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	dtCFL[tx] = C(1e7);

	// no sync here because each tx is unique and a write, nothing is being read from so no risk of trying to access an uninitialised value

	if (x > 0 && x < d_simulationParameters.cells + 1)
	{
		if (d_assembledSolution.h_BC[x] >= d_solverParameters.tol_dry)
		{
			real u = d_assembledSolution.q_BC[x] / d_assembledSolution.h_BC[x];
			dtCFL[tx] = d_solverParameters.CFL * d_dx / (abs(u) + sqrt(d_solverParameters.g * d_assembledSolution.h_BC[x]));
		}
	}

	__syncthreads();

	for (int blockStride = blockDim.x / 2; blockStride > 0; blockStride >>= 1)
	{
		if (tx < blockStride)
		{
			dtCFL[tx] = min(dtCFL[tx], dtCFL[tx + blockStride]);
		}

		__syncthreads(); // same logic as before, sync before next read $(LocalDebuggerCommandArguments)
	}

	if (tx == 0)
	{
		dtCFLblockLevel[blockIdx.x] = dtCFL[0];
	}
}

void checkCUDAError(const char* msg);
int smemPerArray(int threadsPerBlock);

int main()
{
	int test_case = set_test_case();
	int num_cells = set_num_cells();

	clock_t start = clock();

	SimulationParameters h_simulationParameters = set_simulation_parameters(test_case, num_cells);
	SolverParameters     h_solverParameters     = set_solver_parameters();
	BoundaryConditions   h_bcs                  = set_boundary_conditions(test_case);

	cudaMemcpyToSymbol(d_simulationParameters, &h_simulationParameters, sizeof(SimulationParameters));
	checkCUDAError("failed to copy sim params");

	cudaMemcpyToSymbol(d_solverParameters, &h_solverParameters, sizeof(SolverParameters));
	checkCUDAError("failed to copy solver params");

	cudaMemcpyToSymbol(d_bcs, &h_bcs, sizeof(BoundaryConditions));
	checkCUDAError("failed to copy boundary conditions to device");

	real h_dx = (h_simulationParameters.xmax - h_simulationParameters.xmin) / h_simulationParameters.cells;
	cudaMemcpyToSymbol(d_dx, &h_dx, sizeof(real));
	checkCUDAError("failed to copy dx to device");

	int interfaces = h_simulationParameters.cells + 1;
	int sizeInterfaces = interfaces * sizeof(real);

	real* d_xInt;
	real* d_qInt;
	real* d_hInt;
	real* d_zInt;

	cudaMalloc((void**)&d_xInt, sizeInterfaces);
	cudaMalloc((void**)&d_qInt, sizeInterfaces);
	cudaMalloc((void**)&d_hInt, sizeInterfaces);
	cudaMalloc((void**)&d_zInt, sizeInterfaces);
	checkCUDAError("cudaMalloc for interface values failed");

	int threadsPerBlock = 128;
	int numBlocks = (h_simulationParameters.cells + 2) / threadsPerBlock + ((h_simulationParameters.cells + 2) % threadsPerBlock != 0);

	dim3 gridDims(numBlocks); // 1D grid dimensions are simply the number of blocks
	dim3 blockDims(threadsPerBlock);

	meshAndInitialConditions << <gridDims, blockDims >> > (d_xInt, d_zInt, d_hInt, d_qInt, test_case);
	checkCUDAError("kernel meshAndInitialConditions failed");

	cudaDeviceSynchronize();

	int sizeIncBCs = (h_simulationParameters.cells + 2) * sizeof(real);

	AssembledSolution d_assembledSolution;
	
	real* d_etaTemp;

	cudaMalloc((void**)&d_assembledSolution.q_BC, sizeIncBCs);
	cudaMalloc((void**)&d_assembledSolution.h_BC, sizeIncBCs);
	cudaMalloc((void**)&d_assembledSolution.z_BC, sizeIncBCs);
	cudaMalloc((void**)&d_etaTemp, sizeIncBCs);
	checkCUDAError("cudaMalloc for withBC values failed");

	real* checker = (real*)malloc(sizeIncBCs);
	real* checker2 = (real*)malloc(sizeIncBCs);
	real* checker3 = (real*)malloc(sizeIncBCs);
	real* checker4 = (real*)malloc(sizeIncBCs);

	int smemSize = 3 * smemPerArray(threadsPerBlock); // 3 sets of arrays
	modalProjections << <gridDims, blockDims, smemSize >> > (d_qInt, d_hInt, d_zInt, d_assembledSolution);
	checkCUDAError("kernel modalProjections failed");

	cudaDeviceSynchronize();

	int* d_dryCells;
	cudaMalloc((void**)&d_dryCells, (h_simulationParameters.cells + 2) * sizeof(int));
	checkCUDAError("cudaMalloc for dryCells failed");

	int* h_dryCells = (int*)malloc((h_simulationParameters.cells + 2) * sizeof(int));

	FaceValues d_faceValues;

	cudaMalloc((void**)&d_faceValues.qEast, sizeInterfaces);
	cudaMalloc((void**)&d_faceValues.hEast, sizeInterfaces);
	cudaMalloc((void**)&d_faceValues.etaEast, sizeInterfaces);
	checkCUDAError("cudaMalloc for east values failed");

	cudaMalloc((void**)&d_faceValues.qWest, sizeInterfaces);
	cudaMalloc((void**)&d_faceValues.hWest, sizeInterfaces);
	cudaMalloc((void**)&d_faceValues.etaWest, sizeInterfaces);
	checkCUDAError("cudaMalloc for west values failed");

	StarValues d_starValues;

	cudaMalloc((void**)&d_starValues.q_east, sizeInterfaces);
	cudaMalloc((void**)&d_starValues.h_east, sizeInterfaces);
	cudaMalloc((void**)&d_starValues.z_east, sizeInterfaces);
	checkCUDAError("cudaMalloc for east star values failed");

	cudaMalloc((void**)&d_starValues.q_west, sizeInterfaces);
	cudaMalloc((void**)&d_starValues.h_west, sizeInterfaces);
	cudaMalloc((void**)&d_starValues.z_west, sizeInterfaces);
	checkCUDAError("cudaMalloc for west star values failed");

	Fluxes d_fluxes;
	cudaMalloc((void**)&d_fluxes.mass, sizeInterfaces);
	cudaMalloc((void**)&d_fluxes.momentum, sizeInterfaces);

	BarValues d_barValues;
	cudaMalloc((void**)&d_barValues.h, h_simulationParameters.cells * sizeof(real));
	cudaMalloc((void**)&d_barValues.z, h_simulationParameters.cells * sizeof(real));

	real* d_dtCFLblockLevel;
	cudaMalloc((void**)&d_dtCFLblockLevel, numBlocks * sizeof(real));
	real* h_dtCFLblockLevel = (real*)malloc(numBlocks * sizeof(real));

	real timeNow = 0;
	real h_dt = C(1e-3);

	int steps = 0;

	std::ofstream data;

	data.open("clock_time_vs_sim_time.csv");

	data << "timeNow,elapsed" << std::endl;

	while (timeNow < h_simulationParameters.simulationTime)
	{
		timeNow += h_dt;

		if (timeNow - h_simulationParameters.simulationTime > 0)
		{
			timeNow -= h_dt;
			h_dt = h_simulationParameters.simulationTime - timeNow;
			timeNow += h_dt;
		}

		addGhostBCs << < gridDims, blockDims >> > (d_assembledSolution);
		checkCUDAError("kernel addGhostBCs failed");

		cudaDeviceSynchronize();

		initialiseEtaTemp << < gridDims, blockDims >> > (d_assembledSolution, d_etaTemp);
		checkCUDAError("kernel initialiseEtaTemp failed");

		cudaDeviceSynchronize();

		cudaMemcpyToSymbol(d_dt, &h_dt, sizeof(real));
		checkCUDAError("copying time step failed");

		if (h_simulationParameters.manning > 0)
		{
			frictionImplicit << < gridDims, blockDims >> > (d_assembledSolution);
			checkCUDAError("kernel frictionImplicit failed");

			cudaDeviceSynchronize();
		}

		smemSize = smemPerArray(threadsPerBlock);
		wetDryCells << < gridDims, blockDims, smemSize >> > (d_assembledSolution, d_dryCells);
		checkCUDAError("kernel wetDryCells failed");

		cudaDeviceSynchronize();

		smemSize = 3 * smemPerArray(threadsPerBlock);
		initialiseFaceValues << <gridDims, blockDims, smemSize >> > (d_assembledSolution, d_etaTemp, d_faceValues);
		checkCUDAError("kernel initialiseInterfaceValues failed");

		cudaDeviceSynchronize();

		wettingAndDrying << <gridDims, blockDims >> > (d_faceValues, d_starValues);
		checkCUDAError("kernel wettingAndDrying failed");

		cudaDeviceSynchronize();

		fluxHLL << <gridDims, blockDims >> > (d_faceValues, d_starValues, d_fluxes);
		checkCUDAError("kernel fluxHLL failed");

		cudaDeviceSynchronize();

		smemSize = smemPerArray(threadsPerBlock);
		initialiseBarValues << <gridDims, blockDims, smemSize >> > (d_starValues, d_barValues);
		checkCUDAError("kernel barValues failed");
		
		smemSize = 2 * smemPerArray(threadsPerBlock);
		fv1Operator << <gridDims, blockDims, smemSize >> > (d_dryCells, d_barValues, d_fluxes, d_assembledSolution);
		checkCUDAError("kernel fv1Operator failed");

		cudaDeviceSynchronize();

		smemSize = smemPerArray(threadsPerBlock);
		timeStepAdjustment << <gridDims, blockDims, smemSize >> > (d_assembledSolution, d_dtCFLblockLevel);
		checkCUDAError("kernel timeStepAdjustment failed");

		cudaDeviceSynchronize();

		cudaMemcpy(h_dtCFLblockLevel, d_dtCFLblockLevel, numBlocks * sizeof(real), cudaMemcpyDeviceToHost);
		checkCUDAError("failed to copy dtCFL values to host");

		h_dt = h_dtCFLblockLevel[0];

		for (int i = 1; i < numBlocks; i++)
		{
			h_dt = min(h_dtCFLblockLevel[i], h_dt);
		}

		printf("%f s\n", timeNow);

		if (++steps % 100 == 0)
		{
			clock_t end = clock();
			real time = (real)(end - start) / CLOCKS_PER_SEC * C(1000.0);

			data << timeNow << "," << time << std::endl;
		}

	}

	clock_t end = clock();
	real time = (real)(end - start) / CLOCKS_PER_SEC * C(1000.0);

	data << timeNow << "," << time << std::endl;

	data.close();

	cudaMemcpy(checker, d_assembledSolution.q_BC, sizeIncBCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker2, d_assembledSolution.h_BC, sizeIncBCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker3, d_assembledSolution.z_BC, sizeIncBCs, cudaMemcpyDeviceToHost);
	cudaMemcpy(checker4, d_xInt, sizeInterfaces, cudaMemcpyDeviceToHost);

	data.open("debug.csv");

	data << "x,q,eta,z" << std::endl;

	for (int i = 0; i < h_simulationParameters.cells; i++)
	{
		data << (checker4[i] + checker4[i + 1]) / 2 << "," << checker[i + 1] << "," << max(checker2[i + 1] + checker3[i + 1], checker3[i + 1]) << "," << checker3[i + 1] << std::endl;
	}

	data.close();
	
	cudaFree(d_xInt);
	cudaFree(d_qInt);
	cudaFree(d_hInt);
	cudaFree(d_zInt);

	cudaFree(d_assembledSolution.q_BC);
	cudaFree(d_assembledSolution.h_BC);
	cudaFree(d_assembledSolution.z_BC);
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

	end = clock();

	time = (real)(end - start) / CLOCKS_PER_SEC * C(1000.0);
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