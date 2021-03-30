#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <mpi.h>
#include <stdio.h>
#define ITERACOES 200
#define SIZE 1024
#define TAM 512


__global__ void MatrixProblem(float *matriz, float *matrizResultado)
{
    // int tid = threadIdx.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid/TAM > 0 && tid/TAM < TAM-1 && tid%TAM != 0 && tid%TAM != TAM-1){
    	matrizResultado[tid] = (0.25*(matriz[tid+1] + matriz[tid-1] + matriz[tid-TAM] + matriz[tid + TAM] - (4*matriz[tid]))) + matriz[tid];
	}
	else{
		matrizResultado[tid] = 100;
	}
}

int main()
{
	float *pMatriz;
	float *pMatrizResultado;
	float mat[TAM][TAM];
	float mCopy[TAM][TAM];
	double t0, t1;

	//preenchendo com 0
	for(int j = 0; j<TAM; j++){
		for(int i = 0; i<TAM; i++){
			mat[i][j] = 0;
			mCopy[i][j] = 0;
		}
	}

	//inicializando valores
	for(int i=0; i<TAM; i++){
		for (int j = 0; j<TAM; j++){
			if(i == 0 || j == 0 || i == TAM-1 || j == TAM-1){
				mat[i][j] = 100;
				mCopy[i][j] = 100;
			}
		}
	}

	cudaMalloc((void **)&pMatriz, sizeof(float) * size_t(TAM*TAM)); 
	cudaMalloc((void **)&pMatrizResultado, sizeof(float) * size_t(TAM*TAM)); 

	t0 = MPI_Wtime();
	for (int i = 0; i < ITERACOES; ++i){
	    cudaMemcpy(pMatriz, mat, sizeof(float) * size_t(TAM*TAM), cudaMemcpyHostToDevice); 

		MatrixProblem <<<256, SIZE>>>(pMatriz, pMatrizResultado);

		cudaDeviceSynchronize();
		cudaMemcpy(mat, pMatrizResultado, sizeof(float) * size_t(TAM*TAM), cudaMemcpyDeviceToHost);
	}
		
	t1 = MPI_Wtime();

	printf("O processo terminou em %f segundos\n", t1-t0);
    // cout << "terminou em " << t1-t0 << " segundos" << endl;

	// for(int j = 0; j<TAM; j++){
	// 	for(int i = 0; i<TAM; i++){
	// 		printf("%f ", mat[j][i]);
	// 	}
	// 	printf("\n");
	// }

	cudaFree(pMatriz);
	cudaFree(pMatrizResultado);

	return 0;
}
