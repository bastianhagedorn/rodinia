#ifndef OPENCL_HELPER_LIBRARY_H
#define OPENCL_HELPER_LIBRARY_H

#include <CL/cl.h>

#include <stdio.h>
#include <sys/time.h>


// Function prototypes
void fatal_CL(cl_int error, int line_no);


#endif
