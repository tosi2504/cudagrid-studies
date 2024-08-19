#include <Grid/Grid.h>

#include <Grid/threads/Accelerator.h>
#include <Grid/threads/Threads.h>
#include <iostream>
#include <vector>

using namespace Grid;

int main (int argc, char * argv[]) {
    // FILE: Threads.h
    // FILE: Threads.h
    // FILE: Threads.h
#ifdef GRID_OMP // gets defined through compilation (via _OPENMP)
    std::cout << "GRID_OMP defined" << std::endl;
#endif

    thread_for(i, 100, {
        thread_critical {
            std::cout << thread_num() << "/" << thread_max() << std::endl;
        }
    });

    // FILE: ThreadReduction.h
    // FILE: ThreadReduction.h
    // FILE: ThreadReduction.h
    Grid_init(&argc, &argv);
    std::cout << "GridThread::_threads" << GridThread::_threads << std::endl;
    std::cout << "GridThread::_hyperthreads" << GridThread::_hyperthreads << std::endl;
    std::cout << "GridThread::_cores" << GridThread::_cores << std::endl;

    // GetWork
    int mywork, myoff;
    GridThread::GetWork(100, 15, mywork, myoff);
    std::cout << "GridThread::GetWork(nwork = 100, me = 15, mywork, myoff) : mywork =" << mywork << ", myoff=" << myoff << std::endl;

    // Thread sum
    std::cout << "thread_max=" << thread_max() << std::endl;
    std::vector<vRealF> temp(thread_max());
    vRealF resVal = 1;
    thread_region {
        vRealF val = 1;
        GridThread::ThreadSum(temp, val, thread_num());
        if (thread_num() == 0) {
            resVal = val;
        }
    }
    std::cout << resVal << std::endl;


    // FILE: Accelerator.h
    // FILE: Accelerator.h
    // FILE: Accelerator.h
    int someTempVal = 3;
    std::cout << "acceleratorSIMTlane() in host context: " << acceleratorSIMTlane(0) << std::endl;
    std::cout << "acceleratorThreads() = " << acceleratorThreads() << std::endl;
    accelerator_for(i, 1, 32, { // give me 32*32 threads (blocksize = 32*2, numBlocks = 32/2 = 16 -> totalNumOfThreads = 32*2*16 = 32*32)
        printf("acceleratorSIMTlane() in device context: %d\n", acceleratorSIMTlane(0));
        printf("someTempVal in device context: %d\n", someTempVal); // lambda capture copies stuff from the stack to device memory????
    })
    
    // memory stuff
    std::cout << "Memory stuff: Device memory" << std::endl;
    int * h_array = new int[32];
    int * d_array = (int*) acceleratorAllocDevice(sizeof(int)*32);
    for (unsigned i = 0; i < 32; i++) {
        h_array[i] = i;
    }
    acceleratorCopyToDevice(h_array, d_array, sizeof(int)*32);
    accelerator_for(i, 1, 32, {
        printf("d_array[%u]=%d\n", acceleratorSIMTlane(0), d_array[acceleratorSIMTlane(0)]);
    })

    std::cout << "Memory stuff: Shared memory" << std::endl;
    int * u_array = (int*) acceleratorAllocShared(sizeof(int)*32);
    for(unsigned i = 0; i < 32; i++) {
        u_array[i] = 100*i;
    }
    accelerator_for(i, 1, 32, {
        printf("u_array[%u]=%d\n", acceleratorSIMTlane(0), u_array[acceleratorSIMTlane(0)]);
    })
}
