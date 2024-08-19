#include <Grid/Grid.h>
#include <Grid/allocator/MemoryManager.h>

using namespace Grid;

int main (int argc, char * argv[]) {
    Grid_init(&argc, &argv);

    // FILE: Memory Stats
    // FILE: Memory Stats
    // FILE: Memory Stats
    // Whatever, not interesting
    
    // FILE: MemoryManager.h
#ifdef ALLOCATION_CACHE
    std::cout << "ALLOCATION_CACHE defined" << std::endl;
#endif
#ifdef GRID_UVM
    std::cout << "GRID_UVM defined" << std::endl;
#endif
#ifdef GRID_UVM
    std::cout << "GRID_UVM defined" << std::endl;
#endif
    MemoryManager::PrintBytes();
    unsigned numBytes = sizeof(int)*1000000;
    int * u_array = (int*) MemoryManager::SharedAllocate(numBytes);
    for (unsigned i = 0; i < 1000000; i++) {
        u_array[i] = 123;
    }
    MemoryManager::PrintBytes();
    MemoryManager::SharedFree(u_array, numBytes);
    MemoryManager::PrintBytes();

    // FILE: MemoryManagerShared.cc
    // FILE: MemoryManagerShared.cc
    // FILE: MemoryManagerShared.cc
    // ViewClose, ViewOpen, IsOpen do kinda nuffin.
    
    // FILE: AlignedAllocatior.h
    // FILE: AlignedAllocatior.h
    // FILE: AlignedAllocatior.h
    deviceVector<int> devVec(1000000);
    stencilVector<int> stencilVec(3000000);
    Vector<int> uvmVec(2000000);
    MemoryManager::PrintBytes();
}
