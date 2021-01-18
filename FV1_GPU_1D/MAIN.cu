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

// Kernels
#include "get_mesh_and_nodal_values.cuh"
#include "get_modal_values.cuh"
#include "add_ghost_cells.cuh"
#include "init_eta_temp.cuh"
#include "friction_update.cuh"
#include "get_wet_dry_cells.cuh"
#include "get_face_values.cuh"
#include "get_positivity_preserving_nodal_values.cuh"
#include "get_bar_values.cuh"
#include "fluxHLL.cuh"
#include "fv1_operator.cuh"
#include "get_CFL_time_step.cuh"

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

	NodalValues        d_nodal_vals;
	AssembledSolution  d_assem_sol;
	FaceValues         d_face_vals;
	BarValues          d_bar_vals;
	StarValues         d_star_vals;
	Fluxes             d_fluxes;

	// Variables
	real dx = (sim_params.xmax - sim_params.xmin) / sim_params.cells;

	int interfaces = sim_params.cells + 1;
	int sizeInterfaces = interfaces * sizeof(real);

	int threads_per_block = 128;
	int num_blocks = (sim_params.cells + 2) / threads_per_block + ((sim_params.cells + 2) % threads_per_block != 0);

	int num_bytes_shared_memory = 0;

	real timeNow = 0;
	real dt = C(1e-3);

	int steps = 0;

	int sample_rate = 100;

	// Bytesizes
	size_t bytes_real_BCs = (sim_params.cells + 2) * sizeof(real);
	size_t bytes_int_BCs  = (sim_params.cells + 2) * sizeof(int);
	size_t bytes_CFL      = num_blocks * sizeof(real);

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
	real* height = (real*)malloc(bytes_real_BCs);
	real* topo = (real*)malloc(bytes_real_BCs);
	real* x_nodes = (real*)malloc(bytes_real_BCs);
	
	// ============================================================ //
	
	get_mesh_and_nodal_values<<<num_blocks, threads_per_block>>>
	(
		sim_params, 
		bcs, 
		d_nodal_vals, 
		dx, 
		test_case
	);

	num_bytes_shared_memory = 3 * smemPerArray(threads_per_block); // 3 sets of arrays
	get_modal_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>
	(
		sim_params, 
		d_nodal_vals, 
		d_assem_sol
	);

	std::ofstream data;

	// mesh data
	data.open("x.csv");
	cudaMemcpy(x_nodes, d_nodal_vals.x, sizeInterfaces, cudaMemcpyDeviceToHost);
	//data << steps << ",";
	for (int i = 0; i < sim_params.cells; i++) data << (x_nodes[i] + x_nodes[i + 1]) / 2 << "\n";
	data.close();

	// topo data
	data.open("z.csv");
	cudaMemcpy(topo, d_assem_sol.z_BC, bytes_real_BCs, cudaMemcpyDeviceToHost);
	//data << steps << ",";
	for (int i = 0; i < sim_params.cells; i++) data << topo[i + 1] << "\n";
	data.close();

	// water depth, record every sample_rate time steps
	data.open("eta.csv");
	cudaMemcpy(height, d_assem_sol.h_BC, bytes_real_BCs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < sim_params.cells; i++) data << height[i + 1] << ",";
	data << "\n";

	while (timeNow < sim_params.simulationTime)
	{
		timeNow += dt;

		if (timeNow - sim_params.simulationTime > 0)
		{
			timeNow -= dt;
			dt = sim_params.simulationTime - timeNow;
			timeNow += dt;
		}

		add_ghost_cells<<<num_blocks, threads_per_block>>>
		(
			bcs, 
			sim_params, 
			d_assem_sol
		);
		
		init_eta_temp<<<num_blocks, threads_per_block>>>
		(
			sim_params, 
			d_assem_sol, 
			d_eta_temp
		);

		if (sim_params.manning > 0)
		{
			friction_update<<<num_blocks, threads_per_block>>>
			(
				sim_params, 
				solver_params, 
				dt, 
				d_assem_sol
			);
		}

		num_bytes_shared_memory = smemPerArray(threads_per_block);
		get_wet_dry_cells<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>
		(
			sim_params, 
			solver_params, 
			d_assem_sol, 
			d_dry_cells
		);

		num_bytes_shared_memory = 3 * smemPerArray(threads_per_block);
		get_face_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>
		(
			sim_params,
			d_assem_sol, 
			d_eta_temp, 
			d_face_vals
		);
		
		get_positivity_preserving_nodal_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>
		(
			sim_params, 
			solver_params, 
			d_face_vals, 
			d_star_vals
		);
		
		fluxHLL<<<num_blocks, threads_per_block>>>
		(
			sim_params, 
			solver_params, 
			d_face_vals, 
			d_star_vals, 
			d_fluxes
		);
		
		num_bytes_shared_memory = smemPerArray(threads_per_block);
		get_bar_values<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>
		(
			sim_params,
			d_star_vals, 
			d_bar_vals
		);
		
		num_bytes_shared_memory = 2 * smemPerArray(threads_per_block);
		fv1_operator<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>
		(
			sim_params, 
			solver_params, 
			dx, 
			dt, 
			d_dry_cells, 
			d_bar_vals, 
			d_fluxes, 
			d_assem_sol
		);

		num_bytes_shared_memory = smemPerArray(threads_per_block);
		get_CFL_time_step<<<num_blocks, threads_per_block, num_bytes_shared_memory>>>
		(
			sim_params, 
			solver_params, 
			dx, 
			d_assem_sol, 
			d_dtCFLblockLevel
		);
		
		cudaMemcpy(h_dtCFLblockLevel, d_dtCFLblockLevel, num_blocks * sizeof(real), cudaMemcpyDeviceToHost);

		dt = h_dtCFLblockLevel[0];

		for (int i = 1; i < num_blocks; i++) dt = min(h_dtCFLblockLevel[i], dt);

		printf("%f s\n", timeNow);

		if (steps % sample_rate == 0)
		{
			cudaMemcpy(height, d_assem_sol.h_BC, bytes_real_BCs, cudaMemcpyDeviceToHost);
			for (int i = 0; i < sim_params.cells; i++) data << height[i + 1] << ",";
			data << "\n";
		}
	}

	cudaMemcpy(height, d_assem_sol.h_BC, bytes_real_BCs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < sim_params.cells; i++) data << height[i + 1] << ",";
	data << "\n";
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
	free(height);
	free(topo);
	free(x_nodes);

	cudaFree(d_dry_cells);
	free(h_dry_cells);

	cudaFree(d_dtCFLblockLevel);
	free(h_dtCFLblockLevel);

	// =================== //

	clock_t end = clock();

	real time = (real)(end - start) / CLOCKS_PER_SEC * C(1000.0);
	printf("Execution time measured using clock(): %f ms\n", time);

	return 0;
}

int smemPerArray(int threads_per_block) { return threads_per_block * sizeof(real); }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               