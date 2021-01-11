#include "free_nodal_values.cuh"

__host__ void free_nodal_values(NodalValues& d_nodal_values)
{
	free_device(d_nodal_values.h);
	free_device(d_nodal_values.q);
	free_device(d_nodal_values.z);
}