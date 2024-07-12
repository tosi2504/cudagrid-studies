#include "datatypes.h" 
#include "matrix.h"
#include "stopwatch.h"
#include "kernels.h"

constexpr unsigned N = 32;
constexpr unsigned numMatrices = 2000;
// using T = realF;
using T = complexF;
constexpr unsigned numThreads = 256;
constexpr unsigned reps = 100;

int main () {
    MatrixBatch<T,N> Xs(numMatrices), Ys(numMatrices), As(numMatrices);
    Xs.fill_random(0);
    Xs.upload();
    As.fill_random(1);
    As.upload();

    unsigned long micros = 0;
    for (unsigned rep = 0; rep < reps; rep++) {
        stopwatch.reset();
        repeatedMatmul::naive 
            <T, N>
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
