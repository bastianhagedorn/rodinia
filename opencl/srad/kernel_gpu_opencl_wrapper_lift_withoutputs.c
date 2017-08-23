//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h" // (in the main program folder)

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>  // (in path known to compiler)	needed by printf
#include <string.h> // (in path known to compiler)	needed by strlen

#include <CL/cl.h> // (in path specified to compiler)			needed by OpenCL types and functions

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/opencl/opencl.h" // (in directory)							needed by device functions

#include "./../util/timer/timer.h"       // (in specified path)
//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_opencl_wrapper_lift.h" // (in directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200




void kernel_gpu_opencl_wrapper_lift(fp *image,  // input image
                               int Nr,     // IMAGE nbr of rows
                               int Nc,     // IMAGE nbr of cols
                               long Ne,    // IMAGE nbr of elem
                               int niter,  // nbr of iterations
                               fp lambda,  // update step size
                               long NeROI, // ROI nbr of elements
                               int *iN, int *iS, int *jE, int *jW,
                               int iter, // primary loop
                               int mem_size_i, int mem_size_j) {

  //======================================================================================================================================================150
  //	GPU SETUP
  //======================================================================================================================================================150

  //====================================================================================================100
  //	COMMON VARIABLES
  //====================================================================================================100

  // common variables
  cl_int error;

  cl_event event1;
  cl_event event2;

  cl_event srad2CalcLiftEvents[niter];
  cl_event srad2InpLiftEvents[niter];

  //====================================================================================================100
  //	GET PLATFORMS (Intel, AMD, NVIDIA, based on provided library), SELECT
  //ONE
  //====================================================================================================100

  // Get the number of available platforms
  cl_uint num_platforms;
  error = clGetPlatformIDs(0, NULL, &num_platforms);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  // Get the list of available platforms
  cl_platform_id *platforms =
      (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  error = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  // Select the 1st platform
  cl_platform_id platform = platforms[0];

  // Get the name of the selected platform and print it (if there are multiple
  // platforms, choose the first one)
  char pbuf[100];
  error =
      clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  printf("Platform: %s\n", pbuf);

  //====================================================================================================100
  //	CREATE CONTEXT FOR THE PLATFORM
  //====================================================================================================100

  // Create context properties for selected platform
  cl_context_properties context_properties[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

  // Create context for selected platform being GPU
  cl_context context;
  context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_ALL,
                                    NULL, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //====================================================================================================100
  //	GET DEVICES AVAILABLE FOR THE CONTEXT, SELECT ONE
  //====================================================================================================100

  // Get the number of devices (previousely selected for the context)
  size_t devices_size;
  error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  // Get the list of devices (previousely selected for the context)
  cl_device_id *devices = (cl_device_id *)malloc(devices_size);
  error = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, devices,
                           NULL);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  // Select the first device (previousely selected for the context) (if there
  // are multiple devices, choose the first one)
  cl_device_id device;
  device = devices[0];

  // Get the name of the selected device (previousely selected for the context)
  // and print it
  error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(pbuf), pbuf, NULL);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  printf("Device: %s\n", pbuf);

  //====================================================================================================100
  //	CREATE COMMAND QUEUE FOR THE DEVICE
  //====================================================================================================100

  // Create a command queue
  cl_command_queue command_queue;
  command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //====================================================================================================100
  //	CREATE PROGRAM, COMPILE IT
  //====================================================================================================100

  // Load kernel source code from file
  const char *source = load_kernel_source("./kernel/kernel_gpu_opencl_lift.cl");
  size_t sourceSize = strlen(source);

    size_t global_work_size_lift[2];
    global_work_size_lift[0] = 512;//Nr ; 
    global_work_size_lift[1] =512; //Nc; 

    size_t local_work_size_lift[2];
    local_work_size_lift[0] = 128; 
    local_work_size_lift[1] = 2; 


  // Create the program
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, &sourceSize, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  // Compile the program
  error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  // Print warnings and errors from compilation
  static char log[65536];
  memset(log, 0, sizeof(log));
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1,
                        log, NULL);
  printf("-----OpenCL Compiler Output-----\n");
  if (strstr(log, "warning:") || strstr(log, "error:"))
    printf("<<<<\n%s\n>>>>\n", log);
  printf("--------------------------------\n");
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //====================================================================================================100
  //	CREATE Kernels
  //====================================================================================================100

  cl_kernel srad_kernel_lift1;
  srad_kernel_lift1 = clCreateKernel(program, "srad_kernel_lift1", &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);


  cl_kernel srad_kernel_lift2inp;
  srad_kernel_lift2inp = clCreateKernel(program, "srad_kernel_lift2inp", &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);


  int mem_size; // matrix memory size
  mem_size =
      sizeof(fp) * Ne; // get the size of float representation of input IMAGE

  //====================================================================================================100
  // allocate memory for entire IMAGE on DEVICE
  //====================================================================================================100

  cl_mem d_I;
  d_I = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);


  // lift outputs
  cl_mem d_Coeffs;
  d_Coeffs = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_ImageOutput;
  d_ImageOutput= clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //====================================================================================================100
  // allocate memory for derivatives
  //====================================================================================================100

  cl_mem d_dN;
  d_dN = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_dS;
  d_dS = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_dW;
  d_dW = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_dE;
  d_dE = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //====================================================================================================100
  // allocate memory for coefficient on DEVICE
  //====================================================================================================100

  cl_mem d_c;
  d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //======================================================================================================================================================150
  // 	COPY INPUT TO CPU
  //======================================================================================================================================================150

  //====================================================================================================100
  // Image
  //====================================================================================================100

    fp *imageReadIn = (fp *)calloc(Ne, sizeof(fp));
    fp *coeffsReadIn = (fp *)calloc(Ne, sizeof(fp));
    fp *dDNReadIn  = (fp *)calloc(Ne, sizeof(fp));
    fp *dDSReadIn = (fp *)calloc(Ne, sizeof(fp));
    fp *dDEReadIn = (fp *)calloc(Ne, sizeof(fp));
    fp *dDWReadIn = (fp *)calloc(Ne, sizeof(fp));


    readFloatsFromFile(imageReadIn, "orgdata/imagebefore.txt",Ne);
    readFloatsFromFile(coeffsReadIn, "orgdata/coeffs.txt",Ne);
    readFloatsFromFile(dDNReadIn, "orgdata/dDN.txt",Ne);
    readFloatsFromFile(dDSReadIn, "orgdata/dDS.txt",Ne);
    readFloatsFromFile(dDEReadIn, "orgdata/dDE.txt",Ne);
    readFloatsFromFile(dDWReadIn, "orgdata/dDW.txt",Ne);

  error =
      clEnqueueWriteBuffer(command_queue, d_I, 1, 0, mem_size, imageReadIn, 0, 0, 0);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  error =
      clEnqueueWriteBuffer(command_queue, d_c, 1, 0, mem_size, coeffsReadIn, 0, 0, 0);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  error =
      clEnqueueWriteBuffer(command_queue, d_dN, 1, 0, mem_size, dDNReadIn, 0, 0, 0);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  error =
      clEnqueueWriteBuffer(command_queue, d_dS, 1, 0, mem_size, dDSReadIn, 0, 0, 0);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  error =
      clEnqueueWriteBuffer(command_queue, d_dE, 1, 0, mem_size, dDEReadIn, 0, 0, 0);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  error =
      clEnqueueWriteBuffer(command_queue, d_dW, 1, 0, mem_size, dDWReadIn, 0, 0, 0);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

    free(imageReadIn);
    free(coeffsReadIn);
    free(dDNReadIn);
    free(dDSReadIn);
    free(dDEReadIn);
    free(dDWReadIn);

  //======================================================================================================================================================150
  // 	KERNEL EXECUTION PARAMETERS
  //======================================================================================================================================================150

  //====================================================================================================100
  //	SRAD Kernel
  //====================================================================================================100

  //====================================================================================================100
  //	SRAD Kernel LIFT 1
  //====================================================================================================100
  float q0sqr = 0.053787220269; // this value is dependent on data set size !!

  error = clSetKernelArg(srad_kernel_lift1, 0, sizeof(cl_mem), (void *)&d_I);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift1, 1, sizeof(float), (void *)&q0sqr);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift1, 2, sizeof(cl_mem), (void *)&d_Coeffs);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift1, 3, sizeof(int), (void *)&Nr);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift1, 4, sizeof(int), (void *)&Nc);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //====================================================================================================100
  //	SRAD Kernel Lift 2 INP
  //====================================================================================================100
  
  error = clSetKernelArg(srad_kernel_lift2inp, 0, sizeof(cl_mem), (void *)&d_I);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 1, sizeof(cl_mem), (void *)&d_c);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 2, sizeof(cl_mem), (void *)&d_dN);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 3, sizeof(cl_mem), (void *)&d_dS);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 4, sizeof(cl_mem), (void *)&d_dE);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 5, sizeof(cl_mem), (void *)&d_dW);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 6, sizeof(cl_mem), (void *)&d_ImageOutput);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 7, sizeof(int), (void *)&Nr);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clSetKernelArg(srad_kernel_lift2inp, 8, sizeof(int), (void *)&Nc);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  //====================================================================================================100
  //	End
  //====================================================================================================100

  //======================================================================================================================================================150
  // 	COMPUTATION
  //======================================================================================================================================================150

  printf("Iterations Progress: ");

  // execute main loop
  for (iter = 0; iter < niter;
       iter++) { // do for the number of iterations input parameter

   // printf("%d ", iter);
    fflush(NULL);

    /**************** SRAD 1 **************/


    error = clEnqueueNDRangeKernel(command_queue, srad_kernel_lift1, 2, NULL,
                                   global_work_size_lift, local_work_size_lift, 0, NULL,
                                   &event1);

    if (error != CL_SUCCESS)
      fatal_CL(error, __LINE__);


    

    error = clEnqueueNDRangeKernel(command_queue, srad_kernel_lift2inp, 2, NULL,
                                   global_work_size_lift, local_work_size_lift, 0, NULL,
                                   &event2);
    if (error != CL_SUCCESS)
      fatal_CL(error, __LINE__);

  }

  printf("\n");

  //====================================================================================================100
  // synchronize
  //====================================================================================================100

  error = clFinish(command_queue);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //====================================================================================================100
  //	End
  //====================================================================================================100

  //======================================================================================================================================================150
  // 	COPY RESULTS BACK TO CPU
  //======================================================================================================================================================150

  error = clEnqueueReadBuffer(command_queue, d_I, CL_TRUE, 0, mem_size, image,
                              0, NULL, NULL);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //======================================================================================================================================================150
  // 	FREE MEMORY
  //======================================================================================================================================================150
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double kernel1Time = (time_end-time_start);///1000000000.0;//getTimeForAllEvents(niter,srad1Events);
    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernel2Time = (time_end-time_start);///1000000000.0;//getTimeForAllEvents(niter,srad2Events);


/*
    double kernel1Time = getTimeForAllEvents(niter,srad1LiftEvents);
    double kernel2Time = getTimeForAllEvents(niter,srad2InpLiftEvents);

*/
    printf("SRAD1: %f SRAD2: %f\n",kernel1Time/1000000000.0,kernel2Time/1000000000.0);

  // OpenCL structures
  error = clReleaseKernel(srad_kernel_lift1);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseKernel(srad_kernel_lift2inp);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseProgram(program);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  // common_change
  error = clReleaseMemObject(d_I);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseMemObject(d_c);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

// lift specific
  error = clReleaseMemObject(d_ImageOutput);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseMemObject(d_Coeffs);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  error = clReleaseMemObject(d_dN);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseMemObject(d_dS);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseMemObject(d_dE);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseMemObject(d_dW);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  // OpenCL structures
  error = clFlush(command_queue);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseCommandQueue(command_queue);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  error = clReleaseContext(context);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  //======================================================================================================================================================150
  // 	End
  //======================================================================================================================================================150
}
