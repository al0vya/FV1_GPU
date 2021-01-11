#include "free_face_values.cuh"

__host__ void free_face_values(FaceValues& d_face_vals)
{
	free_device(d_face_vals.q_east  );
	free_device(d_face_vals.q_west  );
	free_device(d_face_vals.eta_east);
	free_device(d_face_vals.eta_west);
	free_device(d_face_vals.h_east  );
	free_device(d_face_vals.h_west  );
}