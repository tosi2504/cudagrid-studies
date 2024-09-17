# Running MultiGeneralCoarsenedMatrix

## Compiling
This is weird. When I compile Grid, it does not add the headers for MultiGeneralCoarsenedMatrix.
Maybe this is enabled by some setting during configuration. Let me check.
Okay, so I checked and it seems that the headers ARE installed.
This is different to my notebook ... need to install again maybe.

## Template parameters:
`MultiGeneralCoarsenedMatrix<class Fobj, class CComplex, int nbasis>`
It is instantiated as (e.g.)
MultiGeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis>

### Fobj
This is supposed to have to do something with the fine grid.
It is certainly a tensor type, e.g. vSpinColourVector.
Which in turn is a iScalar<iVector<iVector<vComplex, Nc>, Ns>>.

### CComplex
This is also a tensor type, e.g. vTComplex{,F,D}.
Which in turn is a iScalar<iScalar<iScalar<vComplex{,F,D}>>>.
I have no idea why he does it this why.
But it is very clear that all this is trimmed towards a field theory (3 tensor levels).

SComplex (= CComplex::scalar_object) then is just a iScalar<Complex{,F,D}> (sweet!).

### nbasis
The number of dimensions on the tensor object living on a coarse lattice site.

## Constructor parameters
The constructor must be called correctly, such that my personal stencil operation aligns with this stuff here.
Good source: tests/debug/Test_general_coarse_hdcg_phys48_mixed.cc :)

### geom: NonLocalStencilGeometry
This should describe what points we want in our stencil.
This interpretation is underlined by the reoccuring for-loop over geom.npoint elements.

TODO

### CoarseGridMulti: GridCartesian *
This should be the coarse lattice we are working on.
This might be 5 dimensional to deal with the batch index.
Or maybe not, because it also needs to fit for the matrix field?
We will see.

TODO
