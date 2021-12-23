#include <stdio.h>
#include <stdlib.h>

__global__ void kernel ( double * a, double * b, double * c, int N)
{
  int g_block_i = blockIdx.x * gridDim.y + blockIdx.y;
  int n_threads = blockDim.x * blockDim.y;
  int g_thread_i = g_block_i * n_threads + (threadIdx.y * blockDim.x + threadIdx.x);
  int g_mat_i = g_thread_i / N;
  int g_mat_j = g_thread_i % N;
  int mat_index = g_mat_i * N + g_mat_j;
  double res = 0.0;
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

  cudaMemcpy ( d_a, h_a, size_n, cudaMemcpyHostToDevice);
  cudaMemcpy ( d_b, h_b, size_n, cudaMemcpyHostToDevice);

  dim3 dimBlock ( 8, 8);
  dim3 dimGrid ( N/dimBlock.x, N/dimBlock.y);

  kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

  cudaMemcpy ( h_c, d_c, size_n, cudaMemcpyDeviceToHost);



  for	( int i = 0; i < NN; i++)
  {
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
