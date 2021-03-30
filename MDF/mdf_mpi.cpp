#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <mpi.h>
#define TAM 8
#define TEMP 100
#define MASTER 0
#define TPROCESS 1
#define TIMER 6

using namespace std;

float* method1(size_t inicio, size_t fim, float* t, float* t1);
float* method2(size_t inicio, size_t fim, float* t, float* t1);
float* build_matrix();
void print_matrix(float* mat);

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

void print_matrix(float* mat){
	for(int i=0; i<TAM; i++){
		for(int j=0; j<TAM; j++){
			int k = (i*TAM)+j;
			cout << fixed << setprecision(1) << " " << mat[k];
		}
		cout << endl;
	}
}

int main(int argc, char **argv) {
    int id_task, //Número do processo
		num_task, //Threads
		namelen, //Nome da máquina que o processador está sendo usado
		elmt_task, //Quantidade de elementos na comunicação
		tag_task, //Identificador da comunicação
		work_tasks, //Divisão da matriz por threads
    mdftime; //Número de iterações

    mdftime = TIMER;
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

    while(mdftime){
      if(id_task == MASTER){
        size_t temp = work_tasks;
        temp *= TAM;

        if(mdftime < TIMER){
          for (size_t i = 1; i < TAM-1; i++) {
						cout << m0[temp+i] << endl;
            MPI_Recv(&m0[temp+i], 1, MPI_FLOAT, id_task+1, TPROCESS, MPI_COMM_WORLD, &status);
          }
        }

        m0 = method1(1, work_tasks+1, m0, m1);
        cout << "-------PRINT PROCESSO " << id_task << " ITERAÇÃO " << mdftime << "-------" << endl;
        print_matrix(m0);

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
          MPI_Recv(&m0[temp+i], 1, MPI_FLOAT, id_task-1, TPROCESS, MPI_COMM_WORLD, &status);
        }

        m0 = method1((work_tasks*id_task), (work_tasks*(id_task+1)), m0, m1);

        cout << "-------PRINT PROCESSO " << id_task << " ITERAÇÃO " << mdftime << "-------" << endl;
        print_matrix(m0);

        temp = work_tasks;
        temp *= TAM;
        for (size_t i = 1; i < TAM-1; i++) {
          MPI_Send(&m0[temp+i], 1, MPI_FLOAT, id_task-1, TPROCESS, MPI_COMM_WORLD);
        }

        mdftime--;
      }
    }

    free(m0);
    free(m1);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
