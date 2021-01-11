#pragma once

#include "cuda_runtime.h"

#include "CHECK_CUDA_ERROR.cuh"

void sync();

void peek();

void reset();

void copy
(
	void* dst,
	void* src,
	size_t bytes
);

void* malloc_device
(
	size_t bytes
);

void free_device
(
	void* ptr
);