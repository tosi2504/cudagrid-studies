#include <Grid/Grid.h>

using namespace Grid;

int main (int argc, char * argv[]) {
    Grid_init(&argc, &argv);


    // FILE: SharedMemory.h
    // FILE: SharedMemory.h
    // FILE: SharedMemory.h
#ifdef GRID_COMMS_NONE
    std::cout << "GRID_COMMS_NONE defined" << std::endl;
#endif
#ifdef GRID_COMMS_MPI3
    std::cout << "GRID_COMMS_MPI3 defined" << std::endl;
#endif

    std::cout << "ShmSetup()=" << GlobalSharedMemory::ShmSetup() << std::endl;
    std::cout << "ShmAlloc()=" << GlobalSharedMemory::ShmAlloc() << std::endl;
    std::cout << "ShmAllocBytes()=" << GlobalSharedMemory::ShmAllocBytes() << std::endl;

    
}
