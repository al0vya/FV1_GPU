#include "cuda_utils.cuh"

void sync()
{
	CHECK_CUDA_ERROR( cudaDeviceSynchronize() );
}

void peek()
{
	CHECK_CUDA_ERROR( cudaPeekAtLastError() );
}

void reset()
{
	CHECK_CUDA_ERROR( cudaDeviceReset() );
}

void copy
(
	void* dst,
	void* src,
	size_t bytes
)
{
	CHECK_CUDA_ERROR( cudaMemcpy
	(
		dst,
		src,
		bytes,
		cudaMemcpyDefault
	) );
}

void* malloc_device
(
	size_t bytes
)
{
	void* ptr;
	
	CHECK_CUDA_ERROR( cudaMalloc
	(
		&ptr, 
		bytes
	) );

	return ptr;
}

void free_device
(
	void* ptr
)
{
	CHECK_CUDA_ERROR( cudaFree(ptr) );
}