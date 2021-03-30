#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <mpi.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define TAM 512
#define TEMP 100
#define MASTER 0
#define TPROCESS 1
#define TIMER 200

using namespace std;

__global__ void MatrixProblem(float *matriz, float *matrizResultado)
{
  // int tid = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < TAM*TAM/2){
      if(tid/TAM > 0 && tid/TAM < TAM-1 && tid%TAM != 0 && tid%TAM != TAM-1){
        matrizResultado[tid] = (0.25*(matriz[tid+1] + matriz[tid-1] + matriz[tid-TAM] + matriz[tid + TAM] - (4*matriz[tid]))) + matriz[tid];
    }
    else{
      matrizResultado[tid] = 100;
    }
  }
}

__global__ void MatrixProblem2(float *matriz, float *matrizResultado)
{
  // int tid = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid >= TAM*TAM/2 && tid <= TAM*TAM){
      if(tid/TAM > 0 && tid/TAM < TAM-1 && tid%TAM != 0 && tid%TAM != TAM-1){
        matrizResultado[tid] = (0.25*(matriz[tid+1] + matriz[tid-1] + matriz[tid-TAM] + matriz[tid + TAM] - (4*matriz[tid]))) + matriz[tid];
    }
    else{
      matrizResultado[tid] = 100;
    }
  }
}

#define CHECK_ERROR(call) do {                                                    \
   if( cudaSuccess != call) {                                                             \
      std::cerr << std::endl << "CUDA ERRO: " <<                             \
         cudaGetErrorString(call) <<  " in file: " << __FILE__                \
         << " in line: " << __LINE__ << std::endl;                               \
         exit(0);                                                                                 \
   } } while (0)


float* method1(size_t inicio, size_t fim, float* t, float* t1);
float* method2(size_t inicio, size_t fim, float* t, float* t1);
float* build_matrix();
void print_matrix(float* mat);

void print_matrix(float* mat){
  for(int i=0; i<TAM; i++){
    for(int j=0; j<TAM; j++){
      int k = (i*TAM)+j;
      cout << fixed << setprecision(1) << " " << mat[k];
    }
    cout << endl;
  }
}

float* build_matrix(){
  float* mat = (float*)malloc(TAM * TAM * sizeof(float));

  if(!mat){
    cout << "Espaço para alocação de memória insuficiente" << endl;
    exit(1);
  }

  for(int i=0; i<TAM; i++){
    for(int j=0; j<TAM; j++){
      int k = (i*TAM)+j;
      if(i == 0 || j == 0 || i == TAM-1 || j == TAM-1){
        mat[k] = TEMP;
      } else{
        mat[k] = 0;
      }
    }
  }

  return mat;
}

float* method1(size_t inicio, size_t fim, float* t, float* t1){
    for(int i=inicio; i<fim-1; i++){
      for(int j=1; j<TAM-1; j++){
        t1[(i*TAM)+j] = 0.25 * ( t[((i-1)*TAM)+j] +
        t[((i+1)*TAM)+j] +
        t[(i*TAM)+j-1] +
        t[(i*TAM)+j+1] -
        (4 * t[(i*TAM)+j]) )
        + t[(i*TAM)+j];
      }
    }

  return t1;
}

float* method2(size_t inicio, size_t fim, float* t, float* t1){
  for(int i=inicio; i<fim-1; i++){
    for(int j=1; j<TAM-1; j++){
      t1[(i*TAM)+j] = ( (4 * t[((i-1)*TAM)+j] +
              t[((i+1)*TAM)+j] +
              t[(i*TAM)+j-1] +
              t[(i*TAM)+j+1]) +
              ( t[((i+1)*TAM)+j+1] +
              t[((i+1)*TAM)+j-1] +
              t[((i-1)*TAM)+j-1] +
              t[((i-1)*TAM)+j+1] ) )/20;
     }
   }

  return t1;
}

int main(int argc, char **argv) {
    int id_task, //Número do processo
    num_task, //Threads
    namelen, //Nome da máquina que o processador está sendo usado
    elmt_task, //Quantidade de elementos na comunicação
    tag_task, //Identificador da comunicação
    work_tasks, //Divisão da matriz por threads
    mdftime, //Número de iterações
    timertotal;

    float *pMatriz;
    float *pMatrizResultado;
    float *pMatriz2;
    float *pMatrizResultado2;

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    mdftime = TIMER;
    timertotal = mdftime;
    float* m0 = build_matrix();
    float* m1 = build_matrix();

    double t0, t1; //Tempo de início de fim

    MPI_Status status;

    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_task);
    MPI_Comm_rank(MPI_COMM_WORLD,&id_task);
    MPI_Get_processor_name(processor_name,&namelen);

    work_tasks = TAM/num_task;

    // cout << timertotal << endl;

    cudaMalloc((void **)&pMatriz, sizeof(float) * size_t(TAM*TAM)); 
    cudaMalloc((void **)&pMatrizResultado, sizeof(float) * size_t(TAM*TAM)); 
    cudaMalloc((void **)&pMatriz2, sizeof(float) * size_t(TAM*TAM)); 
    cudaMalloc((void **)&pMatrizResultado2, sizeof(float) * size_t(TAM*TAM)); 

    t0 = MPI_Wtime();

    while(mdftime){
      if(id_task == MASTER){
        size_t temp = work_tasks;
        temp *= TAM;

        if(mdftime < TIMER){
          for (size_t i = 1; i < TAM-1; i++) {
            MPI_Recv(&m0[temp+i], 1, MPI_FLOAT, id_task+1, TPROCESS, MPI_COMM_WORLD, &status);
          }
        }

        // print_matrix(m0);
        cudaMemcpyAsync(pMatriz, m0, sizeof(float) * size_t(TAM*TAM), cudaMemcpyHostToDevice, streams[0]); 

        MatrixProblem <<<256, 1024,0, streams[0]>>>(pMatriz, pMatrizResultado);
        cudaDeviceSynchronize();
        cudaMemcpyAsync(m0, pMatrizResultado, sizeof(float) * size_t(TAM*TAM), cudaMemcpyDeviceToHost, streams[0]);

        // cout << "-------PRINT PROCESSO " << id_task << " ITERAÇÃO " << timertotal - mdftime + 1 << "-------" << endl;
        // print_matrix(m0);

        //Envia a borda para o próximo processo
        temp = work_tasks-1;
        temp *= TAM;
        for (size_t i = 1; i < TAM-1; i++) {
          MPI_Send(&m0[temp+i], 1, MPI_FLOAT, id_task+1, TPROCESS, MPI_COMM_WORLD);
        }

        mdftime--;
      } else{
        //Recebe a borda do processo anterior
        size_t temp = work_tasks-1;
        temp *= TAM;

        for (size_t i = 1; i < TAM-1; i++) {
          if(timertotal - mdftime + 1 > 1){
          MPI_Recv(&m0[temp+i], 1, MPI_FLOAT, id_task-1, TPROCESS, MPI_COMM_WORLD, &status);
            // printf("%f\n", m0[temp+i]);
            // printf("%d\n", timertotal - mdftime + 1);        
          }
        }
    
        cudaMemcpyAsync(pMatriz2, m0, sizeof(float) * size_t(TAM*TAM), cudaMemcpyHostToDevice, streams[1]); 
        MatrixProblem2 <<<256, 1024, 0, streams[1]>>>(pMatriz2, pMatrizResultado2);
        cudaDeviceSynchronize();
        cudaMemcpyAsync(m0, pMatrizResultado2, sizeof(float) * size_t(TAM*TAM), cudaMemcpyDeviceToHost, streams[1]);

        // cout << "-------PRINT PROCESSO " << id_task << " ITERAÇÃO " << timertotal - mdftime + 1 << "-------" << endl;
        // print_matrix(m0);

        temp = work_tasks;
        temp *= TAM;
        for (size_t i = 1; i < TAM-1; i++) {
          MPI_Send(&m0[temp+i], 1, MPI_FLOAT, id_task-1, TPROCESS, MPI_COMM_WORLD);
        }

        mdftime--;
      }
    }

    t1 = MPI_Wtime();

    cout << "terminou em " << t1-t0 << " segundos" << endl;

    if(id_task == MASTER){
        // print_matrix(m0);
    }
    else{
        // print_matrix(m0);
    }

    free(m0);
    free(m1);
    cudaStreamDestroy(streams[0]); 
    cudaStreamDestroy(streams[1]); 
    cudaFree(pMatriz);
    cudaFree(pMatrizResultado);
    cudaFree(pMatriz2);
    cudaFree(pMatrizResultado2);
    MPI_Finalize();

    return EXIT_SUCCESS;
}