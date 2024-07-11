#pragma once

namespace copy {
    template<class T>
    __global__ void naive(T * d_out, const T * d_in, unsigned N) {
        const unsigned i_start = threadIdx.x + blockDim.x*blockIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;
        for (unsigned i = i_start; i < N*N; i += stride) {
            d_out[i] = d_in[i];
        }
    }

    template<class T>
    double calcBW(unsigned N, unsigned reps, unsigned long micros) {
        return sizeof(T) * (long)2*N*N * (long)reps / (double)micros;
    }
}


namespace transpose {
    template<class T>
    __global__ void naive(T * d_out, const T * d_in, unsigned N) {
        // get global index
        const unsigned i_start = threadIdx.x + blockDim.x*blockIdx.x;
        const unsigned stride = blockDim.x * gridDim.x;
        for (unsigned i = i_start; i < N*N; i+= stride) {
            const unsigned x = i%N;
            const unsigned y = i/N;
            d_out[x*N + y] = d_in[y*N + x];
        }
    }

    // assert(numThreads == TN*TN)
    template<class T, unsigned TN>
    __global__ void shmem(T * d_out, const T * d_in, unsigned N) {
        const unsigned numTilesPerSide = N / TN;
        const unsigned xx = (blockIdx.x%numTilesPerSide) * TN;
        const unsigned yy = (blockIdx.x/numTilesPerSide) * TN;
        const unsigned x  = threadIdx.x % TN;
        const unsigned y  = threadIdx.x / TN;

        // shmem
        __shared__ T tile[TN*TN];
        tile[y*TN + x] = d_in[(yy+y)*N + (xx + x)]; // coalesced gmem access
        __syncthreads();
        d_out[(xx+y)*N + (yy+x)] = tile[x*TN + y];  // coalesced gmem access, I thought lol wtf
    }

    template<class T>
    double calcBW(unsigned N, unsigned reps, unsigned long micros) {
        return sizeof(T) * (long)2*N*N * (long)reps / (double)micros;
    }

    template<class T>
    void test(const T * h_out, const T * h_in, unsigned N) {
        for (unsigned x = 0; x < N; x++) {
            for (unsigned y = 0; y < N; y++) {
                if (not isEqual(h_out[y*N+x], h_in[x*N+y])) {
                    std::cout << "Test transpose: Error at x: " << x << ", y: " << y << std::endl;
                    return;
                }
            }
        }
        std::cout << "Test transpose: Test successful" << std::endl;
    }
}

