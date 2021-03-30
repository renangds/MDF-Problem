CC=/usr/local/cuda/bin/nvcc
CFLAGS= -I/usr/lib64/openmpi/1.2.7-gcc/include -I/usr/local/cuda/include -Xcompiler -fopenmp
LDFLAGS= -L/usr/lib64/openmpi/1.2.7-gcc/lib -L/usr/local/cuda/lib
LIB= -lgomp -lcudart -lmpi
SOURCES= mdf_mpi.cu
EXECNAME= mdfmpiexec

all:

mdf:
	mpic++ mdf_mpi.cpp -o mdf_mpi
	mpirun -np 2 mdf_mpi > log.txt

incuda:
	$(CC) -o $(EXECNAME) $(SOURCES) $(LIB) $(LDFLAGS) $(CFLAGS)

clean:
	rm *.o *.linkinfo
