#include "malloc_face_values.cuh"

__host__ void malloc_face_values
(
	FaceValues& d_face_vals,
	int&        count
)
{
	size_t bytes = count * sizeof(real);
	
	d_face_vals.q_east   = (real*)malloc_device(bytes);
	d_face_vals.q_west   = (real*)malloc_device(bytes);
	d_face_vals.eta_east = (real*)malloc_device(bytes);
	d_face_vals.eta_west = (real*)malloc_device(bytes);
	d_face_vals.h_east   = (real*)malloc_device(bytes);
	d_face_vals.h_west   = (real*)malloc_device(bytes);
}