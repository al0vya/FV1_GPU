#include "get_mesh_and_nodal_values.cuh"

__global__ void get_mesh_and_nodal_values
(
	SimulationParameters sim_params, 
	BoundaryConditions   bcs, 
	NodalValues          d_nodal_vals, 
	real                 dx, 
	int                  test_case
)
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