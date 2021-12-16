#include <stdio.h>
#include <stdlib.h>

int main ( int argc, char ** argv)
{
  int N = (argc < 2) ? 1000 : atoi(argv[1]);
  int size_n = N*N*sizeof(double);
  double *h_a, *h_b, *h_c;
  double *d_a, *d_b, *h_c;

  h_a = (double *) malloc ( N * N * sizeof(double));
  h_b = (double *) malloc ( N * N * sizeof(double));
  h_c = (double *) malloc ( N * N * sizeof(double));

  // Init values
  for ( int i = 0; i < (N*N); i++)
    {
      h_a[i] = 1;
      h_b[i] = 1;
      h_c[i] = 0;
    }
  
  // Q.4 : Allouer 3 vecteurs de tailles NxN sur le GPU
  cudaMalloc ((void **) &d_a, size_n);
  cudaMalloc ((void **) &d_b, size_n);
  cudaMalloc ((void **) &d_c, size_n);


  // Free on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Free main memory
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
