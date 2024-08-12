# TODO

Understand:
    - grid->oSites()
    - autoView
    - coalescedRead
    - coalescedReadPermute
    - coalescedWrite

# Grid optimisations

## Grid benchmark

### Compilation

If grid has been installed, the headers are found under /usr/local/include/Grid and the static lib is under /usr/local/lib/libGrid.a.
One can compile a main.cpp as $g++ main.cpp -o out -l:libGrid.a -fopenmp.

If it has not been installed (e.g. my GPU-1node-build) one can link against the build.
g++ main.cpp -o out -I/home/tobias/phd/Grid/build/Grid -O3 -fno-strict-aliasing -L/home/tobias/phd/Grid/build/Grid/libGrid.a -l:libGrid.a -fopenmp
Interestingly, if I compile with clang I get: "Fatal error: 'omp.h' file not found".
But it works with gcc.

#### For single GPU

The configure command I use is:
../configure --enable-comms=none --enable-accelerator=cuda CXX=nvcc --enable-simd=GPU

The summary is:

```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Summary of configuration for Grid v0.7.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
----- GIT VERSION -------------------------------------
commit: 8a098889
branch: develop
date  : 2024-04-30
----- PLATFORM ----------------------------------------
architecture (build)        : x86_64
os (build)                  : linux-gnu
architecture (target)       : x86_64
os (target)                 : linux-gnu
compiler vendor             : gnu
compiler version            : 
----- BUILD OPTIONS -----------------------------------
Nc                          : 3
SIMD                        : GPU (width= 64)
Threading                   : yes
Acceleration                : cuda
Unified virtual memory      : yes
Communications type         : none
Shared memory allocator     : no
Shared memory mmap path     : /var/lib/hugetlbfs/global/pagesize-2MB/
Default precision           : 
Software FP16 conversion    : yes
RNG choice                  : sitmo
GMP                         : no
LAPACK                      : no
FFTW                        : yes
LIME (ILDG support)         : no
HDF5                        : no
build DOXYGEN documentation : no
Sp2n                        : no
----- BUILD FLAGS -------------------------------------
CXXFLAGS:
    -I/home/tobias/phd/Grid
    -O3
    -Xcompiler
    -fno-strict-aliasing
    --expt-extended-lambda
    --expt-relaxed-constexpr
    -Xcompiler
    -fopenmp
LDFLAGS:
    -L/home/tobias/phd/Grid/build/Grid
    -Xcompiler
    -fopenmp
LIBS:
    -lz
    -lcrypto
    -lfftw3f
    -lfftw3
    -lstdc++
    -lm
    -lcuda
    -lz
-------------------------------------------------------
```

The install can be made with (cuda was not installed globally):

```
sudo env PATH=PATH="/usr/local/cuda-12.3/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH" make install -j 6
```

Compilation of the main.cu be done with:

```
nvcc test-accelerator-for.cu -o out -O3 -l:libGrid.a -Xcompiler -fno-strict-aliasing -Xcompiler -fopenmp --expt-relaxed-constexpr --expt-extended-lambda
```

### Stencil

There is example code for stencils on a color-vector-field, where the stencil operation itself is just a simple add operation between all neighbors and the site.
I need a matmul to a gaugelink field, however - or simpler, just a matmul to a normal gauge-matrix-field.
That is something different, quite fundamentally.
It will still require some kind of stencil object and a halo exchange, but I am very unsure on how to approach this.
I feel like the best way to go about this is to first fully understand how the example works and then try to modify it.
Also, if I learn enough to kind of speak the language, Christoph might be able to help me next Tuesday.

#### Understanding the parts

I need to understand the following symbols:

- SparseMatrixBase<Field> --> DONE
- CartesianStencil<vobj, cobj, Parameters>
  - in file Grid/stencil/Stencil.h
  - vobj (vector_object) is e.g. iScalar<iScalar<iVector<vRealD, 3>>> and is a typedef in Grid::Lattice
  - cobj in this case is the same like vobj, but not sure what is does. Might be short for compute_object
  - the constructor is also weird
    - grid, npoints, directions and distances are clear
    - checkerboard(Grid::Even, what even is that) and Parameters are NOT clear
  - what does Stencil.HaloExchange(_in, Compressor) do?
    - what is the Grid::SimpleCompressor<vobj> thing?
      - in file Grid/stencil/SimpleCompressor.h
      - is an template alias for SimpleCompressorGather<vobj, FaceGatherSimple>
        - WTF is a FaceGatherSimple
    - probably performs the halo exchange on the field _in and compresses that data
    - there needs to be some buffer within the stencil object, which saves the halo
  - what does Stencil.View(Grid::AcceleratorRead) do?
    - where does Grid::AcceleratorRead come from?
      - in file Grid/allocator/MemoryManager.h
      - is an enum of name Grid::ViewMode
    - returns a CartesianStencilView, whatever tf that is.
    - return value later called "st"
      - st has member function st.GetEntry(pytpe, dir, ss), wtf does that do?
  - All the core stuff:
    - grid->oSites();
    - autoView
    - coalescedRead
    - coalescedReadPermute
    - coalescedWrite
    - acceleratorSIMTlane(Nsimd)
      - returns the within-warp index (0-63)
    - accelerator_for(iter, num, nsimd, {...}) compiled for GPU
      - in file Grid/threads/Accelerator.h
      - becomes accelerator_for2dNB(iter, num, iter2, 1, nsimd, {...})
        - num1 = num, num2 = 1, num3 = nsimd;
        - cu_threads = {nsimd        , nthrd , 1}
        - cu_blocks  = {num /+ nthrd , 1     , 1}
        - LambdaApply<<<...>>>(num1, 1, nsimd, lambda);

### Grid memory management
What is generally meant by `vobj * _odata` pointers?
So far this only appeared in the context of LatticeViews.
Probably is a managed pointer to the lattice data on this node.
This implies that a `_odata[i]` expression returns a `vobj`.

#### MemoryManager (allocator/MemoryManager.h)
- enum ViewAdvise {AdviseDefault, AdviseInfrequentUse}
- enum ViewMode {AcceleratorRead, AcceleratorWrite, ...}
- #IFDEF GRID_UVM and #IFNDEF ALLOCATION_CACHE
  - MemoryManager::{Accelerator, Shared}{Allocate, Free} just wrap cudaMalloc{, Managed} and cudaFree
  - MemoryManager::init has no effect
  - ViewClose does nothing, ViewOpen returns CpuPtr

#### Communicator (communicator/communicator.h)
- SharedMemory (communicator/SharedMemory.h)
  - static class GlobalSharedMemory -> makes a trivial call to acceleratorAllocDevice(bytes)
- CartesianCommunicator (communicator/Communicator_base.h)

#### LatticeBase, LatticeAccelerator, LatticeView and Lattice (Lattice_view.h & Lattice_base.h)
LatticeBase is empty class.
template<vobj> LatticeAccelerator : public LatticeBase;
template<vobj> LatticeView : public LatticeAccelerator;
template<vobj> Lattice : public LatticeAccelerator;



#### LatticeView: operator()(size_t i) and operator[](size_t i) (lattice/Lattice_view.h)
- operator()(size_t i) as device function:
  - returns vobj::scalar_type
  - `return coalescedRead(this->_odata[i])`
- operator()(size_t i) as host function:
  - returns vobj & (wtfffffffff, why does it behave completely different on host???)
- operator[](size_t i) as device function:
  - returns vobj &
  - `return this->_odata[i]`

#### autoView(l_v, l, mode) (lattice/Lattice_view.h)
Macro that takes mode and l as arguments and puts something (prop a pointer) into l_v.
It also creates a `ViewCloser<decltype(l_v)> _autoView##l_v(l_v)` object.
This object calls l_v.ViewClose() when l_v runs out of scope (genius!).
The type of l_v is indeed LatticeView<vobj> (see Lattice_bash.h > View method)

#### coalescedRead (tensors/Tensor_SIMT.h)
The definition if `GRID_SIMT` (e.g. device compilation pass of nvcc) is defined:

```
template<class vsimd,IfSimd<vsimd> = 0> accelerator_inline
typename vsimd::scalar_type
coalescedRead(const vsimd & __restrict__ vec,int lane=acceleratorSIMTlane(vsimd::Nsimd()))
{
  typedef typename vsimd::scalar_type S;
  S * __restrict__ p=(S *)&vec;
  return p[lane];
}
```

### Benchmark
