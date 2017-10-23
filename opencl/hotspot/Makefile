include ../../common/make.config

# OpenCL installation path      
# OPENCL_DIR = /home/mwb7w/NVIDIA_GPU_Computing_SDK/
# OPENCL_DIR = /home/mwb7w/ati-stream-sdk-v2.2-lnx64/


CC = gcc
CC_FLAGS = -g -O3 -Wall

EXE = hotspot

$(EXE): hotspot.c hotspot.h   
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $(EXE) -I$(OPENCL_INC) hotspot.c -L$(OPENCL_LIB) -lOpenCL -lm 

clean:
	rm -f $(EXE)
