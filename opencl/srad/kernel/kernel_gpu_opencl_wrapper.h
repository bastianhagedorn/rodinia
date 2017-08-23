#ifdef __cplusplus
extern "C" {
#endif

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER HEADER
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper(	fp* image,											// input image
							int Nr,												// IMAGE nbr of rows
							int Nc,												// IMAGE nbr of cols
							long Ne,											// IMAGE nbr of elem
							int niter,											// nbr of iterations
							fp lambda,											// update step size
							long NeROI,											// ROI nbr of elements
							int* iN,
							int* iS,
							int* jE,
							int* jW,
							int iter,											// primary loop
							int mem_size_i,
							int mem_size_j,
                                                        int platform_idx,
                                                        int device_idx);

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
