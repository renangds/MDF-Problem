#include <iostream>
#include <cstdlib>
#include <iomanip>

#define TAM 256
#define TEMP 100

using namespace std;

double* build_matrix(){
	double* mat = (double*)malloc(TAM * TAM * sizeof(double));

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

double* method1(int time, double* t, double* t1){
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

		double* swap = t1;
		t1 = t;
		t = swap;
		time--;
	}

	return t;
}

double* method2(int time, double* t, double* t1){
	while(time){
		for(int i=1; i<TAM-1; i++){
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

		double* swap = t1;
		t1 = t;
		t = swap;

		time--;
	}

	return t;
}

void print_matrix(double* mat){
	for(int i=0; i<TAM; i++){
		for(int j=0; j<TAM; j++){
			int k = (i*TAM)+j;
			cout << fixed << setprecision(1) << " " << mat[k];
		}
		cout << endl;
	}
}

int main(int argc, char const *argv[]){
	double* t = build_matrix();
	double* t1 = build_matrix();

	double* x = method2(1000, t, t1);

	print_matrix(x);	

	free(t);
	free(t1);

	return 0;
}