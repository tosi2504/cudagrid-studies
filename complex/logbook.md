# Complex Numbers with CUDA

## Copy kernel
Write a simple templated copy kernel and benchmark performance between float and complex<float>.
Then tune until same performance has been achieved.
Interestingly complex and real perform identical and both optimal.
I have decided to dive a bit into PTX, so lets decode the following section:
```
.version 8.3
.target sm_52
.address_size 64



.visible .entry _ZN4copy5naiveIfEEvPT_PKS1_j(       // entry point somehow
.param .u64 _ZN4copy5naiveIfEEvPT_PKS1_j_param_0,   // the three parameters
.param .u64 _ZN4copy5naiveIfEEvPT_PKS1_j_param_1,
.param .u32 _ZN4copy5naiveIfEEvPT_PKS1_j_param_2
)
{

// =============> allocate variables?
.reg .pred %p<3>; // pred ??? 
.reg .f32 %f<2>;  // floats
.reg .b32 %r<11>; // bitsized type
.reg .b64 %rd<8>; // probably addresses



// ============> load params
ld.param.u64 %rd3, [_ZN4copy5naiveIfEEvPT_PKS1_j_param_0]; // out-pointer
ld.param.u64 %rd4, [_ZN4copy5naiveIfEEvPT_PKS1_j_param_1]; // in-pointer
ld.param.u32 %r6, [_ZN4copy5naiveIfEEvPT_PKS1_j_param_2];  // len

// ============> thread indexing stuff
mov.u32 %r7, %ctaid.x;  // blockIdx.x to r7
mov.u32 %r1, %ntid.x;   // numThreads to r1
mov.u32 %r8, %tid.x;    // threadIdx  to r8
mad.lo.s32 %r10, %r1, %r7, %r8; // r10 = r1*r2 + r8 = i_start

setp.ge.u32 %p1, %r10, %r6; // p1 = r10 >= r6
@%p1 bra $L__BB0_3; // if i_start (r10) ge len (r6) jump to end


mov.u32 %r9, %nctaid.x; // r9 = numBlocks
mul.lo.s32 %r3, %r1, %r9; // r3 = r1 * r9 = numThreads * numBlocks

cvta.to.global.u64 %rd1, %rd4; // in pointer into rd1
cvta.to.global.u64 %rd2, %rd3; // outpointer into rd2

// ===========> actual loop
$L__BB0_2:

// ===========> actual reading and writing
mul.wide.u32 %rd5, %r10, 4;
add.s64 %rd6, %rd1, %rd5;
ld.global.f32 %f1, [%rd6];
add.s64 %rd7, %rd2, %rd5;
st.global.f32 [%rd7], %f1;

add.s32 %r10, %r10, %r3;

// ===========> check for loop bounds
setp.lt.u32 %p2, %r10, %r6;
@%p2 bra $L__BB0_2; // jump back if not finished

$L__BB0_3:

ret;

}
```
PTX is actually not that hard to understand, at least for such a simple kernel.
If one does this for complex numbers, the global loads and stores are vectorized (i.e. get a .v2 modifier).

## Transpose kernel
Okay let's get a bit more involved.
Here we will write two kernels:
- A naive version with strided gmem access
- An improved version with shared memory
We will then learn about how the complex numbers behave with the shared memory.
I am looking forward to it.

### Naive version
Performs horrible for both complex and real numbers (as expected).
Reaches only about a tenth of the max bandwidth.

### Shared memory
Okay so the idea is simple:
Load a tile into shared memory and then write it back out.
Both the load and the store can be done as coalesced gmem access, but somehow that did not work at all!
Again, I reached only about a tenth of the max bandwidth.
So wtf is going on here???
