#ifndef HOTSPOT_H
#define HOTSPOT_H


#include <CL/cl.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include "../include/constants.h" 

// OpenCL globals
cl_context context;
cl_command_queue command_queue;
cl_device_id device;
cl_kernel kernel;

void writeoutput(float *, int, int, char *);
void readinput(float *, int, int, char *);
int compute_tran_temp(cl_mem, cl_mem[2], int, int, int, int, int, int, int, int, float *, float *);
void usage(int, char **);
void run(int, char **);
double getTimeForAllEvents(int numEvents, cl_event* events);
void  fatal(char* s);
void fatal_CL(cl_int error, int line_no);
long long get_time(); 
char* load_kernel_source(const char* filename);



#endif
