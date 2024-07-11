#include "errorcheck.h"
#include <random>



#define TILE_DIM 32
#define BLOCK_ROWS 8
#define N (2048)


template<typename T>
__global__ void transposeCoalesced(T * out, const T * in) {
    // shmem - delicious
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // read in
    const unsigned x = threadIdx.x + blockIdx.x*TILE_DIM;
    const unsigned y = threadIdx.y + blockIdx.y*TILE_DIM;
    for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = in[(y+j)*N + x];
    }

    // sync
    __syncthreads();

    // write out
    const unsigned x_out = blockIdx.y*TILE_DIM + threadIdx.x;
    const unsigned y_out = blockIdx.x*TILE_DIM + threadIdx.y;
    for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
        out[(y_out+j)*N + x_out] = tile[threadIdx.x][threadIdx.y + j];
    }
}

template<typename T>
__global__ void transposeShmem(T * out, const T * in) {
    // shmem - delicious
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    // read in
    const unsigned x = threadIdx.x + blockIdx.x*TILE_DIM;
    const unsigned y = threadIdx.y + blockIdx.y*TILE_DIM;
    for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = in[(y+j)*N + x];
    }

    // sync
    __syncthreads();

    // write out
    const unsigned x_out = blockIdx.y*TILE_DIM + threadIdx.x;
    const unsigned y_out = blockIdx.x*TILE_DIM + threadIdx.y;
    for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
        out[(y_out+j)*N + x_out] = tile[threadIdx.x][threadIdx.y + j];
    }
}


using T = float;

int main () {
    // prepare some pointers
    T * d_in, *d_out, *h_in, *h_out;
    h_in = new T[N*N];
    h_out = new T[N*N];
    CCE(  cudaMalloc(&d_in, sizeof(T)*N*N)  );
    CCE(  cudaMalloc(&d_out, sizeof(T)*N*N)  );

    // put random values in it
    std::mt19937 gen(0);
    std::uniform_real_distribution<T> dist(0, 1);
    for (unsigned i = 0; i < N*N; i++) {
        h_in[i] = dist(gen);
    }
    CCE(  cudaMemcpy(d_in, h_in, sizeof(T)*N*N, cudaMemcpyHostToDevice)  );

    // call kernel
    dim3 blockDims = {TILE_DIM, BLOCK_ROWS, 1};
    dim3 gridDims = {(N + TILE_DIM - 1)/TILE_DIM, (N + TILE_DIM - 1)/TILE_DIM, 1};
    std::cout << "grd:" << gridDims.x << ", " << gridDims.y << ", " << gridDims.z << std::endl;
    std::cout << "blk:" << blockDims.x << ", " << blockDims.y << ", " << blockDims.z << std::endl;
    for (unsigned i = 0; i < 100; i++) 
        transposeShmem<T><<<gridDims, blockDims>>>(d_out, d_in);
        CLCE();
        CCE(  cudaDeviceSynchronize()  );
    
    // after kernel 
    CCE(  cudaMemcpy(h_out, d_out, sizeof(T)*N*N, cudaMemcpyDeviceToHost)  );
    delete [] h_in;
    delete [] h_out;
    CCE(  cudaFree(d_in)  );
    CCE(  cudaFree(d_out)  );
}
