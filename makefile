CPP = g++

# Use this for your CUDA programs
NVCC = nvcc

# FLAGS for Linux
CFLAGS = -w -O3

BINS = assignment5

all : $(BINS)

clean :
	rm -f $(BINS)
	rm -f *.o

# Demo program. Add more programs by making entries similar to this
assignment5: assignment5.cu
	${NVCC} $(CFLAGS) -o assignment5 assignment5.cu