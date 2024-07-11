#include "datatypes.h" 
#include "matrix.h"
#include "stopwatch.h"
#include "kernels.h"



constexpr unsigned N = 2048;
using T = realF;
// using T = complexF;
// constexpr unsigned numThreads = 256;
constexpr unsigned reps = 100;
constexpr unsigned TN = 16;

int main () {
    Matrix<T> in(N), out(N);
    in.fill_random(0);
    in.upload();

    unsigned long micros = 0;
    for (unsigned rep = 0; rep < reps; rep++) {
        stopwatch.reset();
        
        // transpose::naive 
        //     <T>
        //     <<< CEIL_DIV(N*N, numThreads) , numThreads >>> 
        //     (out.d_data, in.d_data, N);
        transpose::shmem 
            <T, TN>
            <<< (N*N)/(TN*TN) , TN*TN >>> 
            (out.d_data, in.d_data, N);

        CLCE();
        CCE(cudaDeviceSynchronize());
        stopwatch.press();
        micros += stopwatch.getdiff(0);
    }

    // running test
    out.download();
    std::cout << "Running test:" << std::endl;
    transpose::test(out.h_data, in.h_data, N);
    
    // Calculate bandwidth
    std::cout << "Average runtime (us): " << micros/(double)reps << std::endl;
    std::cout << "BW in MB/s: " << transpose::calcBW<T>(N, reps, micros) << std::endl;
}
