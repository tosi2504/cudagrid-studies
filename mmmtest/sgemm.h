#pragma once

template<class T, uint N, uint blocksize>
__global__ void sgemm_coalesced(T * C, const T * A, const T * B, T alpha, T beta) {
	const uint i = blocksize*blockIdx.x + threadIdx.x/blocksize;
	const uint j = blocksize*blockIdx.y + threadIdx.x%blocksize;
	// j is fast running here in respect to inceasing threadIdx.x
	T tmp = 0.0;
	for (uint k = 0; k < N; k++) {
		tmp += A[i*N + k] * B[k*N + j];
	}
	C[i*N + j] = beta*C[i*N + j] + alpha*tmp;
}

template<class T, uint N, uint blocksize>
__global__ void sgemm_sharedmem(T * C, const T * A, const T * B, T alpha, T beta) {
	const uint i_t = threadIdx.x/blocksize;
	const uint j_t = threadIdx.x%blocksize;
	const uint i = blocksize*blockIdx.x + i_t;
	const uint j = blocksize*blockIdx.y + j_t;

	// statically allocate SMEM
	__shared__ T sA[blocksize*blocksize];
	__shared__ T sB[blocksize*blocksize];

	// okay, so outer loop and inner loop for k
	T tmp = 0.0;
	for (uint kk = 0; kk < N; kk+=blocksize) {
		// load block-matrix into shared memory
		sA[threadIdx.x] = A[i*N + kk+j_t];
		sB[threadIdx.x] = B[(kk+i_t)*N + j];
		__syncthreads();
		for (uint delta_k = 0; delta_k < blocksize; delta_k++) {
			// perform matmul on the block-matrix
			tmp += sA[i_t*blocksize + delta_k]*sB[delta_k*blocksize + j_t];
		}
		__syncthreads();
	}
	C[i*N + j] = alpha*tmp + beta*C[i*N + j];
}

template<class T, uint N, uint blocksize>
__global__ void sgemm_sharedmem_pointerarithm(T * C, const T * A, const T * B, T alpha, T beta) {
	const uint i_t = threadIdx.x/blocksize;
	const uint j_t = threadIdx.x%blocksize;

	// statically allocate SMEM
	__shared__ T sA[blocksize*blocksize];
	__shared__ T sB[blocksize*blocksize];

	// progress A and B to starting points
	A += blocksize*blockIdx.x*N;
	B += blocksize*blockIdx.y;
	C += blocksize*(blockIdx.x*N + blockIdx.y);

	// okay, so outer loop and inner loop for k
	T tmp = 0.0;
	for (uint kk = 0; kk < N; kk+=blocksize) {
		// load block-matrix into shared memory
		sA[threadIdx.x] = A[i_t*N + j_t];
		sB[threadIdx.x] = B[i_t*N + j_t];
		__syncthreads();

		A += blocksize;
		B += blocksize*N;

		for (uint delta_k = 0; delta_k < blocksize; delta_k++) {
			// perform matmul on the block-matrix
			tmp += sA[i_t*blocksize + delta_k]*sB[delta_k*blocksize + j_t];
		}
		__syncthreads();
	}
	C[i_t*N + j_t] = alpha*tmp + beta*C[i_t*N + j_t];
}

// WARNING: 
// assert(blockDim == BN * BK)
// assert(BN % BK == 0)
template<class T, uint N, uint BN, uint BK>
__global__ void sgemm_1D_blocktiling(T * C, const T * A, const T * B, T alpha, T beta) {
	static_assert(BN % BK == 0);
	constexpr uint numThreads = BN * BK;
	constexpr uint TN = BN / BK; // I dont think this is true lol ... wait it is

	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint aRow = threadIdx.x/BK;
	const uint aCol = threadIdx.x%BK;
	
	const uint bRow = threadIdx.x/BN;
	const uint bCol = threadIdx.x%BN;

	const uint threadRow = bRow;
	const uint threadCol = bCol;
	
	// okay lets set the starting points
	C += cRow*BN*N + cCol*BN;
	A += cRow*BN*N;
	B += cCol*BN;

	// statically allocate SMEM
	__shared__ T sA[numThreads];			// this is a BN X BK matrix
	__shared__ T sB[numThreads];			// this is a BK X BN matrix

	// buffer for results -> lies in registries
	T threadResults[TN] = {0.0};

	// lets define the slow-running k-loop 
	for (uint k = 0; k < N; k+=BK) { // K/BK iterations
		// fill shared memory and advance pointers
		sA[threadIdx.x] = A[aRow*N + aCol]; // 1 access to GMEM
		sB[threadIdx.x] = B[bRow*N + bCol]; // 1 access to GMEM
		__syncthreads();
		A += BK;
		B += BK*N;

		// okay what now
		// inner k-loop and loop over threadResults and somehow buffer B
		for (uint delta_k = 0; delta_k < BK; delta_k++) { // BK iterations (8)
			T bTmp = sB[delta_k*BN + threadCol]; // 1 access to SMEM
			for (uint resIdx = 0; resIdx < TN; resIdx++) { // TN iterations (8)
				threadResults[resIdx] += sA[(threadRow*TN + resIdx)*BK + delta_k] * bTmp; // 1 access to SMEM
			}
		}
		__syncthreads();
	}

	for (uint resIdx = 0; resIdx < TN; resIdx++) {
		uint cIdx = (threadRow*TN + resIdx)*N + threadCol;
		C[cIdx] = alpha*threadResults[resIdx] + beta*C[cIdx];
	}
}

template<class T, uint N, uint BN, uint BK, uint TN>
__global__ void sgemm_2D_blocktiling(T * C, const T * A, const T * B, T alpha, T beta) {
	static_assert(BN%TN == 0);
	constexpr uint numThreads = (BN*BN)/(TN*TN); 
	static_assert(numThreads%BK == 0); // for SMEM loading of A: guaranteeing rectangular load tile
	static_assert(numThreads%BN == 0); // for SMEM loading of B
	static_assert((BN*BK)%numThreads == 0); // for SMEM loading: guaranteeing no overloading

	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	constexpr uint offsetStrideA = numThreads/BK;
	constexpr uint offsetStrideB = numThreads/BN;

	const uint threadRow = threadIdx.x / (BN/TN);
	const uint threadCol = threadIdx.x % (BN/TN);

	// advance pointers
	B += cCol*BN;
	A += cRow*BN*N;
	C += cRow*BN*N + cCol*BN;

	// allocate shared memory
	__shared__ T sA[BN*BK];
	__shared__ T sB[BK*BN];

	// allocate result array (in registers)
	T results[TN*TN] = {0.0};

	// allocate register caches
	T regACol[TN];
	T regBRow[TN];

	// outer k loop
	for (uint k = 0; k < N; k+=BK) {
		// need to fill shared memory
		for (uint offset = 0; offset < BN; offset += offsetStrideA) {
			sA[threadIdx.x + offset*BK] = A[(threadIdx.x/BK + offset)*N + threadIdx.x%BK];
		}
		for (uint offset = 0; offset < BK; offset += offsetStrideB) {
			sB[threadIdx.x + offset*BN] = B[(threadIdx.x/BN + offset)*N + threadIdx.x%BN];
		}
		__syncthreads();

		// advance pointers
		A += BK;
		B += BK*N;

		// inner k loop
		for (uint delta_k = 0; delta_k < BK; delta_k++) {
			// populate registries to save smem accesses
			for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
				regACol[resIdxRow] = sA[(threadRow*TN + resIdxRow)*BK + delta_k];
			}
			for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
				regBRow[resIdxCol] = sB[delta_k*BN + threadCol*TN + resIdxCol];
			}
			for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
				for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
					results[resIdxRow*TN+resIdxCol] += regACol[resIdxRow] * regBRow[resIdxCol];
				}
			}
		}
		__syncthreads();
	}
	
	// write results into C
	for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
		for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
			const uint cIdx = (threadRow*TN + resIdxRow)*N + threadCol*TN + resIdxCol;
			C[cIdx] = alpha * results[resIdxRow*TN + resIdxCol] + beta*C[cIdx];
		}
	}
}

template<class Tcomplex, uint N, uint BN, uint BK, uint TN>
__global__ void sgemm_2D_blocktiling_complex(Tcomplex * C, const Tcomplex * A, const Tcomplex * B, Tcomplex alpha, Tcomplex beta) {
    using T = typename Tcomplex::T_base;
	static_assert(BN%TN == 0);
	constexpr uint numThreads = (BN*BN)/(TN*TN); 
	static_assert(numThreads%BK == 0); // for SMEM loading of A: guaranteeing rectangular load tile
	static_assert(numThreads%BN == 0); // for SMEM loading of B
	static_assert((BN*BK)%numThreads == 0); // for SMEM loading: guaranteeing no overloading

	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint threadRow = threadIdx.x / (BN/TN);
	const uint threadCol = threadIdx.x % (BN/TN);

	// advance pointers
	B += cCol*BN;
	A += cRow*BN*N;
	C += cRow*BN*N + cCol*BN;

	// allocate shared memory
	__shared__ T sA[2*BN*BK];
	T * sA_real = sA;
	T * sA_imag = sA + BN*BK;

	__shared__ T sB[2*BK*BN];
	T * sB_real = sB;
	T * sB_imag = sB + BK*BN;

	// allocate result array (in registers)
	T results_real[TN*TN] = {0.0};
	T results_imag[TN*TN] = {0.0};

	// allocate register caches
	T regACol_real[TN];
	T regACol_imag[TN];
	T regBRow_real[TN];
	T regBRow_imag[TN];

	// outer k loop
	for (uint k = 0; k < N; k+=BK) {
        for (uint i = threadIdx.x; i < 2*BK*BN; i+= numThreads) {
            const uint isImag = i & 1;
            const uint col = i % (BK*2);
            const uint col_2 = col >> 1;
            const uint row = i / (BK*2);
            const T * baseA = (T *)A; // since row major -> double the width
            sA[BK*BN*isImag + row*BK + col_2] = baseA[row*2*N + col];
        }
        for (uint i = threadIdx.x; i < 2*BN*BK; i+= numThreads) {
            const uint isImag = i & 1;
            const uint col = i % (BN*2);
            const uint col_2 = col >> 1;
            const uint row = i / (BN*2);
            const T * baseB = (T *)B;
            sB[BN*BK*isImag + row*BN + col_2] = baseB[row*2*N + col];
        }
		__syncthreads();

		// advance pointers
		A += BK;
		B += BK*N;

		// inner k loop
		for (uint delta_k = 0; delta_k < BK; delta_k++) {
			// populate register file to save smem accesses
			for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
				// regACol[resIdxRow] = sA[(threadRow*TN + resIdxRow)*BK + delta_k];
				regACol_real[resIdxRow] = sA_real[(threadRow*TN + resIdxRow)*BK + delta_k];
				regACol_imag[resIdxRow] = sA_imag[(threadRow*TN + resIdxRow)*BK + delta_k];
			}
			for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
				// regBRow[resIdxCol] = sB[delta_k*BN + threadCol*TN + resIdxCol];
				regBRow_real[resIdxCol] = sB_real[delta_k*BN + threadCol*TN + resIdxCol];
				regBRow_imag[resIdxCol] = sB_imag[delta_k*BN + threadCol*TN + resIdxCol];
			}
			for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
				for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
					results_real[resIdxRow*TN+resIdxCol] += 
                        regACol_real[resIdxRow] * regBRow_real[resIdxCol]
                        - regACol_imag[resIdxRow] * regBRow_imag[resIdxCol];
					results_imag[resIdxRow*TN+resIdxCol] += 
                        regACol_real[resIdxRow] * regBRow_imag[resIdxCol]
                        + regACol_imag[resIdxRow] * regBRow_real[resIdxCol];
				}
			}
		}
		__syncthreads();
	}
	
	// write results into C
	for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
		for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
			const uint cIdx = (threadRow*TN + resIdxRow)*N + threadCol*TN + resIdxCol;
            const Tcomplex temp (
                results_real[resIdxRow*TN + resIdxCol],
                results_imag[resIdxRow*TN + resIdxCol]
            );
			C[cIdx] = alpha * temp + beta*C[cIdx];
		}
	}
}

template<class T, uint N, uint BN, uint BK, uint TN>
__global__ void sgemm_vectorized(T * C, const T * A, const T * B, T alpha, T beta) {
	static_assert(BN%TN == 0);
	constexpr uint numThreads = (BN*BN)/(TN*TN); 
	static_assert(numThreads%BK == 0); // for SMEM loading of A: guaranteeing rectangular load tile
	static_assert(numThreads%BN == 0); // for SMEM loading of B
	static_assert((BN*BK)%(numThreads*4) == 0); // for SMEM loading: guaranteeing no overloading
	// "*4" bc each thread now loads float4
	static_assert(BN%4 == 0); // else the vectorized float fails bc it "overshoots"
	static_assert(BK%4 == 0);
	static_assert(TN%4 == 0); // for vectorized stores into C in GMEM from registers

	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	constexpr uint offsetStrideA = (numThreads*4)/BK;
	constexpr uint offsetStrideB = (numThreads*4)/BN;
	// "*4" bc each thread now loads float4

	const uint threadRow = threadIdx.x / (BN/TN);
	const uint threadCol = threadIdx.x % (BN/TN);

	const uint threadIdx4 = threadIdx.x*4;

	// advance pointers
	B += cCol*BN;
	A += cRow*BN*N;
	C += cRow*BN*N + cCol*BN;

	// allocate shared memory
	__shared__ T sA_T[BK*BN]; // sA is now transposed
	__shared__ T sB[BK*BN];

	// allocate result array (in registers)
	T results[TN*TN] = {0.0};

	// allocate register caches
	T regACol[TN];
	T regBRow[TN];

	// outer k loop
	for (uint k = 0; k < N; k+=BK) {
		// need to fill shared memory --> want to vectorize this
		for (uint offset = 0; offset < BN; offset += offsetStrideA) {
			// sA_T[(threadIdx%BK)*BN + threadIdx/BK + offset] = A[(threadIdx/BK + offset)*N + threadIdx%BK];
			float4 tmp = reinterpret_cast<const float4 *>(&A[(threadIdx4/BK + offset)*N + threadIdx4%BK])[0];
			sA_T[((threadIdx4+0)%BK)*BN + (threadIdx4+0)/BK + offset] = tmp.x;
			sA_T[((threadIdx4+1)%BK)*BN + (threadIdx4+1)/BK + offset] = tmp.y;
			sA_T[((threadIdx4+2)%BK)*BN + (threadIdx4+2)/BK + offset] = tmp.z;
			sA_T[((threadIdx4+3)%BK)*BN + (threadIdx4+3)/BK + offset] = tmp.w;
		}
		for (uint offset = 0; offset < BK; offset += offsetStrideB) {
			sB[threadIdx.x + offset*BN] = B[(threadIdx.x/BN + offset)*N + threadIdx.x%BN];
			reinterpret_cast<float4*>(&sB[threadIdx4 + offset*BN])[0]
					= reinterpret_cast<const float4*>(&B[(threadIdx4/BN + offset)*N + threadIdx4%BN])[0];
		}
		__syncthreads();

		// advance pointers
		A += BK;
		B += BK*N;

		// inner k loop
		for (uint delta_k = 0; delta_k < BK; delta_k++) {
			// populate registries to save smem accesses
			for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
				regACol[resIdxRow] = sA_T[delta_k*BN + threadRow*TN + resIdxRow];
			}
			for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
				regBRow[resIdxCol] = sB[delta_k*BN + threadCol*TN + resIdxCol];
			}
			for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
				for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
					results[resIdxRow*TN+resIdxCol] += regACol[resIdxRow] * regBRow[resIdxCol];
				}
			}
		}
		__syncthreads();
	}
	
	// write results into C
	for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
		for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol+=4) { // skip 3 bc of vectorized stores
			const uint cIdx = (threadRow*TN + resIdxRow)*N + threadCol*TN + resIdxCol;
			float4 tmpC = reinterpret_cast<float4*>(&C[cIdx])[0];
			float4 tmpCNew;
			tmpCNew.x = alpha*results[resIdxRow*TN + resIdxCol + 0] + beta*tmpC.x;
			tmpCNew.y = alpha*results[resIdxRow*TN + resIdxCol + 1] + beta*tmpC.y;
			tmpCNew.z = alpha*results[resIdxRow*TN + resIdxCol + 2] + beta*tmpC.z;
			tmpCNew.w = alpha*results[resIdxRow*TN + resIdxCol + 3] + beta*tmpC.w;
			reinterpret_cast<float4*>(&C[cIdx])[0] = tmpCNew;
		}
	}
}
