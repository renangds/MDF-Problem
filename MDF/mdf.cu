#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <ctime>

#define BLOCK_SIZE 32

using namespace std;

#define CHECK_ERROR(call) do {                                         
   if( cudaSuccess != call) {                                          
      std::cerr << std::endl << "CUDA ERRO: " <<                       
         cudaGetErrorString(call) <<  " in file: " << __FILE__         
         << " in line: " << __LINE__ << std::endl;                      
         exit(0);                                                      
	}	} while (0)

__global__ void MyKernel(float* output, const float value){

}


//Primeiro método a ser utilizado de forma sequencial
float* method1(int time, float* t, float* t1){
	while(time){
		for(int i=1; i<TAM-1; i++){
			for(int j=1; j<TAM-1; j++){
				t1[(i*TAM)+j] = 0.25 * ( t[((i-1)*TAM)+j] +
				t[((i+1)*TAM)+j] +
				t[(i*TAM)+j-1] +
				t[(i*TAM)+j+1] -
				(4 * t[(i*TAM)+j]) ) 
				+ t[(i*TAM)+j];
			}
		}

		float* swap = t1;
		t1 = t;
		t = swap;
		time--;
	}

	return t;
}


__global__ void method1(float* mat, float* matres){

}


int main(int argc, char const *argv[])
{
	int tam = (int)atoi(argv[1]);

	float* t = build_matrix(); //Matriz com os valores iniciais
	float* t1 = build_matrix(); //Matriz com os valores atualizados

	cudaEvent_t start; 
	cudaEvent_t stop;  

	double* d_mat1, d_mat2;

	//Alocando espaço de memória
	CHECK_ERROR(cudaMalloc((void**)&d_mat1, size_t(TAM*TAM)*sizeof(double)));
	CHECK_ERROR(cudaMalloc((void**)&d_mat2, size_t(TAM*TAM)*sizeof(double)));

	method1 <<<1, tam>>>(d_mat1, d_mat2);


	//Desalocando memória
	cudaFree(d_mat1);
	cudaFree(d_mat2);

	return 0;
}