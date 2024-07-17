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
Note: On my big machine it was 33%!

### Shared memory
Okay so the idea is simple:
Load a tile into shared memory and then write it back out.
Both the load and the store can be done as coalesced gmem access, but somehow that did not work at all!
Again, I reached only about a tenth of the max bandwidth.
Okay, so on my big machine it works, i.e., we git about 85% of bandwidth with the shmem kernel. 
This means I just simply cannot trust my notebook, when not plugged into power.
Now let's try with complex numbers -> Again, no performance issues (we get 90%).
It should be noted, that the pressure on shared memory is low.
We only have one access per gmem element.
This is hard to compare with matmul operations.
However, we have won some important insight:
Gmem to shmem works very well, independently of complex and real numbers.


## Matmul: High arithmetic density
So far we have looked into memory-heavy operations and found that the switch from real to complex numbers has no implications on optimal code.
Now, I want to do something a bit compute-heavier.
So, here is the idea:
Load small matrices into shmem (1 matrix per block) and perform some kind of matmul on those.
We can crank up the arithmetic density by repeating that operation multiple times.
For now, we just do it once.
As inputs we should have an input field and an output field.
Those should be continuous in memory? -> Does not matter actually.
Should we do the full blocktiling stuff?
Let's not do this now, but instead add it gradually.
Problem: We need to stride through the result matrix somehow.
Solved.
Let's write some test code and a main file.
Alright, test routine is also written ... let's write the main file.
Main file has been written and kernel has been debugged.
The performance results are very interesting!
TODO: 
- check performance for realF and complexF and try to interpret
- implement the iterations loop to play with arithmetic density
- this is gonna be good

### Performance for iterations = 1
Alright, results are in and very interesting:
For 
```
    T           : {realF, complexF}
    N           : 32
    numMatrices : 2000
    numThreads  : 256
    reps        : 100
    shmem(bytes): 16384
```
we get respectively:
- realF: 250 GB/s
- complexF: 342 GB/s

What do we learn from this?
For small matrices the arithmetic density is so low that the bottleneck is the actual gmem loads and stores.
It is surprising though that the bandwidth for complex numbers is higher than for real numbers.
I suspect, that has something to do with vectorized gmem access, but that will require further investigation.
Let's do some profiling.
- real numbers:
    - we already run into MIO Throttle Stalls, most likely due to lower arithmetic density (2 shmemloads(4byte): 2 flop -> 1 flop per 4 byte)
    - only 8% of fp32 performance
    - i.e. we are effectively being stalled by doing shmem loads
- complex numbers:
    - not MIO Throttle Stalls yet, most likely bc arithmetic density per shmem load is high (2 shmemloads(8byte): 8 flop -> 1 flop per 2 byte)
    - 23% of fp32 performance
What I do not get, is how the gmem bandwidth is positively correlated to higher arithmetic density.
The ratio of shmem loads to gmem loads is the same for both number types.
The solution is probably vectorized accesses, which a quick look into PTX verifies.
I am surprised though, that that makes such a difference.
Well, lets keep it in mind.

### Performance for iterations > 1
Very interesting results!
I thought that the bandwidth of real numbers will surpass that of complex numbers.
But interestingly the ratio stays roughly 2:3 ... wtf.
Also the usage of fp32 instructions hasn't changed much (it got slightly higher).
So maybe it is really the shmem accesses that dominate.
This would be backed up by the (now much higher ... even for complex numbers) Stall MIO Throttle values.
I am not completely sure what to make of this.

#### Vectorized shmem access during the dot product
I want to have a look at the vectorization of the shmem accesses.
SUPER INTERESTING:
Let's introduce a new metric: shmem load calls per loaded bytes.
This is a metric that measures the magnitude of vectorization at hand.
For both real and complex numbers the k loop is unrolled.
Real numbers: 
So, for real numbers for 4 results we have one vectorized load from A (ld.shared.v4.f32) and then 4 loads from X (4 x ld.shared.f32).
This is in total 5 calls per 8 * 4bytes = 5 calls / 32 bytes.
For complex numbers for 2 results we have one vectorized load from A (ld.shared.v4.f32) and 2 vectorized loads from X (2 x ld.shared.v2.f32).
This is in total 3 calls per 8 * 4bytes = 3 calls / 32 bytes.
THIS IS THE ANSWER TO THE 2:3 problem!!!!
If my theory is correct, this ratio should hold for higher N.
Let's try N = 64 now.
Aaaaaand it holds perfectly (1054 : 1527)!
I just won so much understanding lol.

##### Vectorising X accesses
Vectorizing sX accesses can be easily done by making sX column major.
This indeed makes the sX accesses vectorized.
Interestingly the performance greatly degrades.
This is due to memory bank conflicts.
Avoiding these by k = (_k + threadIdx.x) % N restores performance but removes vectorization.
Let's try to unroll/vectorize the k loop explicitly.
This was not beneficial for performance!
The vectorized sA accesses are great, bc they can be broadcasted as well.
This is not the case for the vectorized sX accesses.
This leads so huuuuge MIO throttle stalls.

#### Vectorized gmem accesses
One more thing I want to understand, is how much effect the vectorized gmem access have.
The real kernel does not have vectorized gmem accesses, while the complex kernel does.
Whatever.

#### Fused multiply add instructions
The complex kernel does not use FMA instructions.
I think this can be solved rather easily though, with the following code:
```
tempR = fmsub(AR, XR, fmsub(AI, XI, tempR[i]));
tempI[i] = fmadd(AR, XI, fmadd(XR, AI, tempI[i]));
```
We could use the CUDA intrinsics functions.
This actually improved performance by like 25% -> NICE!


#### Controlling memory bank conflicts
So here is the theory:
For 'N=64' complex performs significantly worse than real.
The reason is that one row is distributed over 4 memory bank stacks (1 stack = 32x4 bytes).
This means there are 4 memory bank conflicts per load!
For real numbers its only 2 stacks, which seems to be fine.
Note that the following rows in Y are not a problem bc they lead to broadcasts in X loads.
So in a way we are trying to maximize the number of broadcasts.

What do we do about it?
Okay, so here is what I have done:
Introduced new template parameters tileRowStride and tileColStride.
Then the numThreads must be tileRowStride * tileColStride!
The number of conflicts is (tileColStride * tilewidth {* 2 for complex and * 1 for real}) / 32.

Here are the results for N=64, complex, tilewidth=tileheight=2, tileRowStride=32, tileColStride=16 (i.e. 2 conflicts):
300000 MB/s !!!!!!!!!!!!!!!!!!!!!!
That is another big improvement!
We are in fact getting there.

Further improvement for N=64, complex, tilewidth=1, tileheight=2, tileRowStride=32, tileColStride=32 (i.e. 2 conflicts):
313000 MB/s !!!!!!!!!!!!!!!!!!!!!!

Note however, that for N=32 we already have very good complex performance with just blocktiling.
This can be nicely reproduced with the new kernel, though (tilewidth = 2, tileColStride = 16).

TODOs:
- put new kernel (repeatingMatmul::conflicting) through some more careful profiling (nsight-compute).
- carry the findings over to the stencil :)







