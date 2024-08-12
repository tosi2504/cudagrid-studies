project('grid-test', 'cuda', 'cpp')

# compilers
cuda_compiler = meson.get_compiler('cuda')

# options
add_project_arguments( '-Xcompiler', language: 'cuda')
add_project_arguments( '-fno-strict-aliasing', language: 'cuda')
add_project_arguments( '--expt-relaxed-constexpr', language: 'cuda')
add_project_arguments( '--expt-extended-lambda', language: 'cuda')
# need to manually add this path to nvcc compilation
# Would not be required for cpp file compilation (lol, fck nvidia)
add_project_arguments( '-I/usr/local/include', language: 'cuda') 
add_project_arguments( '-Xcompiler', language: 'cuda')
add_project_arguments( '-fopenmp', language: 'cuda') 


# dependencies
zlib_dep = dependency('zlib') # for -lz
openssl_dep = dependency('openssl') # for -lcrypto
mpfr_dep = dependency('mpfr')
gmp_dep = dependency('gmp')
stdcpp_dep = cuda_compiler.find_library('stdc++')
m_dep = cuda_compiler.find_library('m')
openmp_dep = dependency('openmp')
message('OpenMP C flags: ' + openmp_dep.get_compile_args().to_string())
message('OpenMP linker flags: ' + openmp_dep.get_link_args().to_string())
cuda_dep = dependency('cuda')
grid_dep = cuda_compiler.find_library('Grid')

dependencies = [
  zlib_dep, 
  openssl_dep, 
  mpfr_dep,
  gmp_dep, 
  stdcpp_dep, 
  m_dep, 
  cuda_dep, 
  grid_dep
]


# executables
exe_source_files = {
  'simd': 'tests/simd.cu',
  'comptest': 'tests/comptest.cpp'
}

foreach name, file : exe_source_files
  message('Processing file: ' + file)
  executable(name, sources: file, dependencies: dependencies)
endforeach



