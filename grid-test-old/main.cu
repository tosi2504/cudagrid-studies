#include <Grid/Grid.h>



const std::vector<int> directions   ({Grid::Xdir, Grid::Ydir,Grid::Zdir,Grid::Xdir,Grid::Ydir,Grid::Zdir});
const std::vector<int> displacements({1,1,1,-1,-1,-1});

using namespace Grid;

#define LEG_LOAD(dir)                                   \
    SE = st.GetEntry(ptype, dir, ss);                   \
    if (SE->_is_local) {                                \
        int perm = SE->_permute;                        \
        chi = coalescedReadPermute(in[SE->_offset], ptype, perm, lane); \
    } else {                                            \
        chi = coalescedRead(buf[SE->_offset], lane);    \
    }                                                   \
    acceleratorSynchronise();                           \

typedef Grid::LatticeColourVector Field;
// template<class Field>
class FreeLaplacianStencil : public Grid::SparseMatrixBase<Field> {
    public:
    typedef typename Field::vector_object siteObject;
    typedef Grid::CartesianStencil<siteObject, siteObject, Grid::SimpleStencilParams> StencilImpl;

    Grid::GridBase * grid;
    StencilImpl Stencil;
    Grid::SimpleCompressor<siteObject> Compressor;
    
    FreeLaplacianStencil(Grid::GridBase * _grid):
        Stencil(_grid, 6, Grid::Even, directions, displacements, Grid::SimpleStencilParams()), grid(_grid) {}

    virtual Grid::GridBase * Grid(void) { 
        return grid; 
    }

    virtual void M (const Field & _in, Field & _out) {
        // huh? -> Halo exchange before calculations I suppose?
        Stencil.HaloExchange(_in, Compressor);

        auto st = Stencil.View(Grid::AcceleratorRead);
        auto buf = st.CommBuf();
        autoView(in, _in, Grid::AcceleratorRead);
        autoView(out, _in, Grid::AcceleratorWrite);
        typedef typename Field::vector_object vobj;
        typedef decltype(coalescedRead(in[0])) calcObj;

        const int Nsimd = vobj::Nsimd();
        const uint64_t NN = grid->oSites();
        
        accelerator_for(ss, NN, Nsimd, {
            StencilEntry * SE;
            const int lane = acceleratorSIMTlane(Nsimd);
            calcObj chi;
            calcObj res;
            int ptype;
            res = coalescedRead(in[ss])*(-6.0);
            LEG_LOAD(0); res = res + chi;
            LEG_LOAD(1); res = res + chi;
            LEG_LOAD(2); res = res + chi;
            LEG_LOAD(3); res = res + chi;
            LEG_LOAD(4); res = res + chi;
            LEG_LOAD(5); res = res + chi;
            coalescedWrite(out[ss], res, lane);
        });
    }
    virtual void Mdag (const Field &in, Field &out) {
        M(in, out);
    }
    virtual void Mdiag(const Field &in, Field &out) {
        assert(0);
    }
    virtual void Mdir(const Field &in, Field &out, int dir, int disp) {
        assert(0);
    }
    virtual void MdirAll(const Field &in, std::vector<Field> &out) {
        assert(0);
    }
};


int main (int argc, char * argv[]) {
    Grid::Grid_init(&argc, &argv);
    
    auto latt_size = Grid::GridDefaultLatt();
    auto simd_layout = Grid::GridDefaultSimd(Grid::Nd, Grid::vComplex::Nsimd());
    auto mpi_layout = Grid::GridDefaultMpi();

    std::cout << latt_size << std::endl;

     
    // Grid::LatticeColourVector::vector_object a;

    Grid::GridCartesian grid(latt_size, simd_layout, mpi_layout);
    Grid::GridParallelRNG RNG(&grid);
    RNG.SeedFixedIntegers({45, 12, 81, 9});

    Field in(&grid), out(&grid);
    Grid::gaussian(RNG, in);
    FreeLaplacianStencil<Field> FLst(&grid);
    // const Lattice<iScalar<iScalar<iVector<Grid_simd<complex<double>, vec<double>>, 3>>>> &_in,
    // Lattice<iScalar<iScalar<iVector<Grid_simd<complex<double>, vec<double>>, 3>>>> &_out
    FLst.M(in, out);

    //continue here

    Grid::Grid_finalize();
}
