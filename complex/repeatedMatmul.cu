#include "datatypes.h" 
#include "matrix.h"
#include "stopwatch.h"
#include "kernels.h"

// using T = realF;
using T = complexF;
constexpr unsigned N = 64;
constexpr unsigned numMatrices = 2000;
constexpr unsigned iterations = 200;
constexpr unsigned numThreads = 256;
constexpr unsigned reps = 100;

int main () {
    // print information of run
    std::cout << "repeatedMatmul::naive" << std::endl;
    std::cout << "    T           : " << typeAsString<T>::value << std::endl;
    std::cout << "    N           : " << N << std::endl;
    std::cout << "    iterations  : " << iterations << std::endl;
    std::cout << "    numMatrices : " << numMatrices << std::endl;
    std::cout << "    numThreads  : " << numThreads << std::endl;
    std::cout << "    reps        : " << reps << std::endl;
    std::cout << "    shmem(bytes): " << repeatedMatmul::calcShmemSize<T,N>() << std::endl;

    MatrixBatch<T,N> Xs(numMatrices), Ys(numMatrices), As(numMatrices);
    Xs.fill_random(0);
    Xs.upload();
    As.fill_random(1);
    As.upload();

    unsigned long micros = 0;
    CCE(  cudaFuncSetAttribute(
                repeatedMatmul::naive<T,N,iterations>
                , cudaFuncAttributeMaxDynamicSharedMemorySize
                , repeatedMatmul::calcShmemSize<T,N>()
    )  );
    for (unsigned rep = 0; rep < reps; rep++) {
        stopwatch.reset();
        repeatedMatmul::naive 
            <T, N, iterations>
            <<< numMatrices , numThreads , repeatedMatmul::calcShmemSize<T,N>() >>> 
            (Ys.d_data, As.d_data, Xs.d_data);

        CLCE();
        CCE(cudaDeviceSynchronize());
        stopwatch.press();
        micros += stopwatch.getdiff(0);
    }

    // running test
    Ys.download();
    std::cout << "Running test:" << std::endl;
    repeatedMatmul::test(Ys, As, Xs);
    
    // Calculate bandwidth
    std::cout << "Average runtime (us): " << micros/(double)reps << std::endl;
    std::cout << "BW in MB/s: " << repeatedMatmul::calcBW<T,N>(numMatrices, reps, micros) << std::endl;
}
