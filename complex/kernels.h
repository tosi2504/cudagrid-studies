#pragma once

#include <assert.h>
#include "cudatools.h"
#include "datatypes.h"

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

namespace repeatedMatmul {
    template<class T, unsigned N, unsigned iterations>
    __global__ void naive(T * d_Ys, const T * d_As, const T * d_Xs) {
        const unsigned N_ceil32 = CEIL(N, 32);
        const unsigned numThreads = blockDim.x;
        assert(numThreads >= N_ceil32);

        const unsigned iMatrix = blockIdx.x;
        
        // shift to correct matrix in batch
        const T * A = d_As + N*N*iMatrix;
        const T * X = d_Xs + N*N*iMatrix;
        T * Y = d_Ys + N*N*iMatrix;

        //shmem (everything row major?)
        extern __shared__ T shmem[];
        T * sA = shmem;
        T * sX = shmem + N*N;
        // T * sY = shmem + N*N + N*N;

        // fill sA and sX
        for (unsigned i = threadIdx.x; i < N*N; i+=numThreads) {
            const unsigned col = i % N;
            const unsigned row = i / N;
            sA[i] = A[row*N + col];
            sX[i] = X[row*N + col];
        }
        __syncthreads();

        // iterations loop
        for (unsigned _ = 0; _ < iterations; _++) {
            const unsigned rowStride = numThreads / N_ceil32;
            const unsigned dRow = threadIdx.x / N_ceil32;
            const unsigned iCol = threadIdx.x % N_ceil32;
            if (iCol >= N) return; // access guard
            for (unsigned iiRow = 0; iiRow < N; iiRow+=rowStride) {
                const unsigned iRow = iiRow + dRow;
                if (iRow >= N) break; // access guard
            
                T dotRes = 0;
                for (unsigned k = 0; k < N; k++) {
                    dotRes += sA[iRow*N + k] * sX[k*N + iCol];
                }
                Y[iRow*N + iCol] = dotRes;
            }
        }
    }

    template<class T, unsigned N, unsigned iterations>
    __global__ void vectorizedShmem(T * d_Ys, const T * d_As, const T * d_Xs) {
        const unsigned N_ceil32 = CEIL(N, 32);
        const unsigned numThreads = blockDim.x;
        assert(numThreads >= N_ceil32);

        const unsigned iMatrix = blockIdx.x;
        
        // shift to correct matrix in batch
        const T * A = d_As + N*N*iMatrix;
        const T * X = d_Xs + N*N*iMatrix;
        T * Y = d_Ys + N*N*iMatrix;

        //shmem (everything row major?)
        extern __shared__ T shmem[];
        T * sA = shmem;
        T * sX = shmem + N*N;
        // T * sY = shmem + N*N + N*N;

        // fill sA and sX
        for (unsigned i = threadIdx.x; i < N*N; i+=numThreads) {
            const unsigned col = i % N;
            const unsigned row = i / N;
            sA[i] = A[row*N + col  ]; // rowm
            sX[i] = X[row   + col*N]; // colm
        }
        __syncthreads();

        // iterations loop
        for (unsigned _ = 0; _ < iterations; _++) {
            const unsigned rowStride = numThreads / N_ceil32;
            const unsigned dRow = threadIdx.x / N_ceil32;
            const unsigned iCol = threadIdx.x % N_ceil32;
            if (iCol >= N) return; // access guard
            for (unsigned iiRow = 0; iiRow < N; iiRow+=rowStride) {
                const unsigned iRow = iiRow + dRow;
                if (iRow >= N) break; // access guard
            
                T dotRes = 0;
                // for (unsigned _kk = 0; _kk < N; _kk+= (16/sizeof(T)) ) {
                for (unsigned _kk = 0; _kk < N; _kk += 2) {
                    // const unsigned kk = (_kk + FLOOR(threadIdx.x, 16/sizeof(T)))%N;
                    const unsigned kk = (_kk + FLOOR(threadIdx.x, 2))%N;
                    // for (unsigned dk = 0; dk < 16/sizeof(T); dk++) {
                    for (unsigned dk = 0; dk < 2; dk++) {
                        const unsigned k = kk + dk;
                        dotRes += sA[iRow*N + k] * sX[k + iCol*N];
                    }
                }
                Y[iRow*N + iCol] = dotRes;
            }
        }
    }

     template<class T, unsigned N, unsigned iterations, unsigned tileheight, unsigned tilewidth>
    __global__ void blocktiling(T * d_Ys, const T * d_As, const T * d_Xs) {
        const unsigned numThreads = blockDim.x;

        static_assert(N % tileheight == 0);
        static_assert(N % tilewidth == 0);
        constexpr unsigned numTileCols = N / tilewidth; 
        constexpr unsigned numTileRows = N / tileheight; 
        assert(numThreads*tilewidth >= N); // can fill at least one row with tiles

        const unsigned iMatrix = blockIdx.x;
        
        // shift to correct matrix in batch
        const T * A = d_As + N*N*iMatrix;
        const T * X = d_Xs + N*N*iMatrix;
        T * Y = d_Ys + N*N*iMatrix;

        //shmem (everything row major?)
        extern __shared__ T shmem[];
        T * sA = shmem;
        T * sX = shmem + N*N;
        // T * sY = shmem + N*N + N*N;

        // fill sA and sX
        for (unsigned i = threadIdx.x; i < N*N; i+=numThreads) {
            const unsigned col = i % N;
            const unsigned row = i / N;
            sA[i] = A[row*N + col]; // rowm
            sX[i] = X[row*N + col]; // rowm
        }
        __syncthreads();

        // iterations loop
        const unsigned tileRowStride = numThreads / numTileCols;
        const unsigned iTileCol = threadIdx.x % numTileCols;
        const unsigned iiCol = iTileCol * tilewidth;
        const unsigned dTileRow = threadIdx.x / numTileCols;
        for (unsigned _ = 0; _ < iterations; _++) {
            // need to iterate over rows of A
            for (unsigned iiTileRow = 0; iiTileRow < N; iiTileRow += tileRowStride) {
                const unsigned iTileRow = iiTileRow + dTileRow;
                const unsigned iiRow = iTileRow * tileheight;
                if (iTileRow >= numTileRows) break; // access guard

                // already the k loop?
                T regTile[tileheight][tilewidth] = {0};
                for (unsigned k = 0; k < N; k++) {
                    // need to fill registers
                    T regX[tilewidth] = {0};
                    for(unsigned iRegCol = 0; iRegCol < tilewidth; iRegCol++) {
                        regX[iRegCol] = sX[k*N + iiCol+iRegCol];
                    }

                    T regA[tileheight] = {0};
                    for(unsigned iRegRow = 0; iRegRow < tileheight; iRegRow++) {
                        regA[iRegRow] = sA[(iiRow + iRegRow)*N + k];
                    }

                    // dot product within tile
                    for(unsigned iRegRow = 0; iRegRow < tileheight; iRegRow++) {
                        for(unsigned iRegCol = 0; iRegCol < tilewidth; iRegCol++) {
                            // regTile[iRegRow][iRegCol] += regA[iRegRow] * regX[iRegCol];
                            multiply_accumulate<T>(regTile[iRegRow][iRegCol], regA[iRegRow], regX[iRegCol]);
                        }
                    }
                }
                
                // write out results
                for(unsigned iRegRow = 0; iRegRow < tileheight; iRegRow++) {
                    for(unsigned iRegCol = 0; iRegCol < tilewidth; iRegCol++) {
                        Y[(iiRow + iRegRow)*N + (iiCol + iRegCol)] = regTile[iRegRow][iRegCol];
                    }
                }
            }
        }
    }

     template<class T, unsigned N, unsigned iterations, unsigned tileheight, unsigned tilewidth, unsigned tileRowStride, unsigned tileColStride>
    __global__ void conflicting(T * d_Ys, const T * d_As, const T * d_Xs) {
        const unsigned numThreads = blockDim.x;

        static_assert(N % tileheight == 0);
        static_assert(N % tilewidth == 0);
        constexpr unsigned numTileCols = N / tilewidth;
        constexpr unsigned numTileRows = N / tileheight;
        assert(numThreads == tileColStride*tileRowStride);

        const unsigned iMatrix = blockIdx.x;
        
        // shift to correct matrix in batch
        const T * A = d_As + N*N*iMatrix;
        const T * X = d_Xs + N*N*iMatrix;
        T * Y = d_Ys + N*N*iMatrix;

        //shmem (everything row major?)
        extern __shared__ T shmem[];
        T * sA = shmem;
        T * sX = shmem + N*N;
        // T * sY = shmem + N*N + N*N;

        // fill sA and sX
        for (unsigned i = threadIdx.x; i < N*N; i+=numThreads) {
            const unsigned col = i % N;
            const unsigned row = i / N;
            sA[i] = A[row*N + col]; // rowm
            sX[i] = X[row*N + col]; // rowm
        }
        __syncthreads();

        const unsigned dTileCol = threadIdx.x % tileColStride;
        const unsigned dTileRow = threadIdx.x / tileColStride;

        // need to iterate over rows of A
        for (unsigned iiTileCol = 0; iiTileCol < N; iiTileCol += tileColStride) {
            const unsigned iTileCol = iiTileCol + dTileCol;
            const unsigned iiCol = iTileCol * tilewidth;
            if (iTileCol >= numTileCols) break;
            for (unsigned iiTileRow = 0; iiTileRow < N; iiTileRow += tileRowStride) {
                const unsigned iTileRow = iiTileRow + dTileRow;
                const unsigned iiRow = iTileRow * tileheight;
                if (iTileRow >= numTileRows) break; // access guard

                // already the k loop?
                T regTile[tileheight][tilewidth] = {0};
                for (unsigned k = 0; k < N; k++) {
                    // need to fill registers
                    T regX[tilewidth] = {0};
                    for(unsigned iRegCol = 0; iRegCol < tilewidth; iRegCol++) {
                        regX[iRegCol] = sX[k*N + iiCol+iRegCol]; // no access guard needed bc assert
                    }

                    T regA[tileheight] = {0};
                    for(unsigned iRegRow = 0; iRegRow < tileheight; iRegRow++) {
                        regA[iRegRow] = sA[(iiRow + iRegRow)*N + k]; // no access guard needed bc assert
                    }

                    // dot product within tile
                    for(unsigned iRegRow = 0; iRegRow < tileheight; iRegRow++) {
                        for(unsigned iRegCol = 0; iRegCol < tilewidth; iRegCol++) {
                            // regTile[iRegRow][iRegCol] += regA[iRegRow] * regX[iRegCol];
                            multiply_accumulate<T>(regTile[iRegRow][iRegCol], regA[iRegRow], regX[iRegCol]);
                        }
                    }
                }
                
                // write out results
                for(unsigned iRegRow = 0; iRegRow < tileheight; iRegRow++) {
                    for(unsigned iRegCol = 0; iRegCol < tilewidth; iRegCol++) {
                        Y[(iiRow + iRegRow)*N + (iiCol + iRegCol)] = regTile[iRegRow][iRegCol];
                    }
                }
            }
        }
    }

    template<class T, unsigned N>
    constexpr unsigned calcShmemSize() {
        return sizeof(T) * N*N * 2;
    }

    template<class T, unsigned N>
    double calcBW(unsigned numMatrices, unsigned reps, unsigned long micros) {
        return sizeof(T) * (long)reps * (long)(3*N*N) * (long)(numMatrices) / (double)micros;
    }

    template<class T, unsigned N>
    void test(
        const MatrixBatch<T,N> & Ys, 
        const MatrixBatch<T,N> & As, 
        const MatrixBatch<T,N> & Xs
    ) {
        const unsigned numMatrices = Ys.numMatrices;
        assert(numMatrices == Xs.numMatrices);
        assert(numMatrices == As.numMatrices);
        for (unsigned iMatrix = 0; iMatrix < numMatrices; iMatrix++) {
            T * A = As.h_data + iMatrix*N*N;
            T * X = Xs.h_data + iMatrix*N*N;
            T * Y = Ys.h_data + iMatrix*N*N;
            for (unsigned row = 0; row < N; row++) {
                for (unsigned col = 0; col < N; col++) {
                    T tempRes = 0;
                    for (unsigned k = 0; k < N; k++) {
                        tempRes += A[row*N + k] * X[k*N + col];
                    }
                    if (not isEqual(Y[row*N + col], tempRes)) {
                        std::cout << "Test: Mismatch at iMatrix: " << iMatrix;
                        std::cout << ", row: " << row << ", col:" << col;
                        std::cout << "  --> You suck! *look of disgust and distain*" << std::endl;
                        return;
                    }
                }
            }
        }
        std::cout << "Test: All tests successful, no mismatches! *smirk* *blows kiss*" << std::endl;
    }
}
