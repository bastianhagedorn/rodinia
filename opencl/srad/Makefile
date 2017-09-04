ifndef CC
	CC = gcc
endif

OUTPUT = *.out
CCFLAGS=-O3 -lm 

LIBS = -lOpenCL 
OPENCL_DIR = /home/s1042579/heliumpp/build
OPENCL_INC = $(OPENCL_DIR)/include/ 
OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL
INC=../include/ 

srad: main.c ../include/CL_helper.c \
      kernel/kernel_gpu_opencl_wrapper.c \
      util/graphics/graphics.c \
      util/graphics/resize.c 
	$(CC) $(KERNEL_DIM) $^ $(CCFLAGS) -I$(OPENCL_INC) -I$(INC) -I./kernel $(LIBS) -L$(OPENCL_LIB) -o $@

clean:
	rm -f srad $(OUTPUT) 
