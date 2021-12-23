#include <stdio.h>
#include <stdlib.h>

__global__ void kernel ( double * a, double * b, double * c, int N)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.y * blockDim.x + threadIdx.x, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i,     // Indice global du thread
      g_mat_i    = g_thread_i / N,                         // Indice de la ligne de l'element C a manipuler
      g_mat_j    = g_thread_i % N;                         // Indice de la colonne de l'element C a manipuler

  int mat_index  = g_mat_i * N + g_mat_j;                  // Indice c de l'element C pour une matrice linearisee
  double res = 0.0;

  // Calcul de la valeur de l'element C
  for ( int i = 0; i < N; i++)
  {
    res += a[g_mat_i*N+i] * b[i+g_mat_j*N];
  }
  c[mat_index] = res;
}

int main ( int argc, char ** argv)
{
  int N = (argc < 2) ? 32 : atoi(argv[1]);
  int NN = N*N;
  int size_n = NN*sizeof(double);
  double *h_a, *h_b, *h_c;
  double *d_a, *d_b, *d_c;

  h_a = (double *) malloc ( size_n);
  h_b = (double *) malloc ( size_n);
  h_c = (double *) malloc ( size_n);

  // Init values
  for ( int i = 0; i < NN; i++)
    {
      h_a[i] = 1;
      h_b[i] = 1;
      h_c[i] = 0;
    }
  
  // Q.4 : Allouer 3 vecteurs de tailles NxN sur le GPU
  cudaMalloc ((void **) &d_a, size_n);
  cudaMalloc ((void **) &d_b, size_n);
  cudaMalloc ((void **) &d_c, size_n);

  // Q.5 : Transferer les donnes par l'Host vers le Device
  cudaMemcpy ( d_a, h_a, size_n, cudaMemcpyHostToDevice);
  cudaMemcpy ( d_b, h_b, size_n, cudaMemcpyHostToDevice);

  // Q.6 : Configuration de la grille d'exécution des threads
  dim3 dimBlock ( 32, 32);
  dim3 dimGrid ( N/dimBlock.x, N/dimBlock.y);

  // Q.7 : Execution du kernel de multipilcation matricielle
  kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

  // Q.8 : Transfert des resultats depuis le Devince vers l'Host
  cudaMemcpy ( h_c, d_c, size_n, cudaMemcpyDeviceToHost);


  // Verification de l'exactitude des resultats (matrices a et b tq a(i,j) = b(i,j) = 1
  for	( int i = 0; i < NN; i++)
  {
    // c[i,j] doit etre egal a N
    if ( h_c[i] != N)
    {
      printf ("=error= (%d) -> %lf\n", i, h_c[i]);
    }
  }

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
