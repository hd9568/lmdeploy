#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cstdio>
#include <vector>

namespace turbomind::gemm {

class CublasKernel: public Kernel {
public:
    explicit CublasKernel(): cublas_{}
    {
        cublasCreate(&cublas_);
        if (0) {
            cublasSetMathMode(cublas_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        }

        desc_.backend    = 1;
        desc_.group_axis = -1;

        info_.chunk_size_k      = 1;
        info_.dynamic_smem_size = 0;

        info_.name = GetName();
    }

    ~CublasKernel() override
    {
        cublasDestroy(cublas_);
        cublas_ = {};
    }

    int Launch(const Operation&    operation,
               float               alpha,
               const void*         A,
               const MatrixLayout& Adesc,
               const void*         U,
               const MatrixLayout& Udesc,
               const void*         B,
               const MatrixLayout& Bdesc,
               const void*         V,
               const MatrixLayout& Vdesc,
               float               beta,
               const void*         C,
               const MatrixLayout& Cdesc,
               void*               D,
               const MatrixLayout& Ddesc,
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) override
    {
        cublasOperation_t transa = Adesc.order == kColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t transb = Bdesc.order == kColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;

        const int m = Adesc.rows;
        const int n = Bdesc.cols;
        const int k = Adesc.cols;

        TM_CHECK_EQ(Bdesc.rows, k);
        TM_CHECK_EQ(Ddesc.rows, m);
        TM_CHECK_EQ(Ddesc.cols, n);

        TM_CHECK(C == nullptr || C == D);

        if (stream_ != stream) {
            cublasSetStream(cublas_, stream);
            stream_ = stream;
        }

        if (workspace_ != workspace.partials || workspace_size_ != workspace.partials_size) {
            cublasSetWorkspace(cublas_, workspace.partials, workspace.partials_size);
            workspace_      = workspace.partials;
            workspace_size_ = workspace.partials_size;
        }

        auto ec = cublasGemmEx(cublas_,
                               transa,
                               transb,
                               m,
                               n,
                               k,
                               &alpha,
                               A,
                               to_cuda_dtype(Adesc.type),
                               Adesc.ld,
                               B,
                               to_cuda_dtype(Bdesc.type),
                               Bdesc.ld,
                               &beta,
                               D,
                               to_cuda_dtype(Ddesc.type),
                               Ddesc.ld,
                               CUDA_R_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        return ec == CUBLAS_STATUS_SUCCESS ? 0 : 1;
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        constexpr std::tuple flat3{Striding::kFlat, Striding::kFlat, Striding::kFlat};

        if (std::tie(desc.striding_a, desc.striding_b, desc.striding_c) != flat3) {
            return false;
        }
        if (std::tie(desc.pack_a, desc.pack_b, desc.pack_u, desc.pack_v) != std::tuple{0, 0, 0, 0}) {
            return false;
        }
        if (desc.epilogue != Epilogue::kNone) {
            return false;
        }
        if (desc.num > 1) {
            return false;
        }
        if (desc.quant_a || desc.quant_b) {
            return false;
        }
        if (desc.group_axis >= 0) {
            return false;
        }
        if (desc.order_c != kColMajor) {
            return false;
        }
        if (desc.type_a != kHalf && desc.type_a != kBfloat16 && desc.type_a != kFloat) {
            return false;
        }
        if (desc.type_b != desc.type_a) {
            return false;
        }
        if (desc.type_c != desc.type_a && desc.type_c != kFloat) {
            return false;
        }
        return true;
    }

    int GetMaxSwizzle(const int4&) const override
    {
        return 0;
    }

    int GetMaxSplits(const int4&, int, size_t, size_t) const override
    {
        return 1;
    }

private:
    cublasHandle_t cublas_{};
    cudaStream_t   stream_{};
    void*          workspace_{};
    size_t         workspace_size_{};
};

void Registry::cublas_float()
{
    Add(std::make_unique<CublasKernel>());
}

#if defined(ENABLE_CUBLAS_GROUPED)

// Grouped GEMM via cublasGemmGroupedBatchedEx (CUDA 12.5+, SM100).
// Requires standard (K,N) row-major weight; GetConverters skips tiled conversion on SM100 for grouped BF16.
// Problem (row-major): D_i = A_i * B_i^T  with A_i (M_i,K), B_i (N,K), D_i (M_i,N).
// cuBLAS (col-major):  C = alpha*op(A)*op(B) + beta*C.
// Map: C_cublas = D^T (N,M_i), A_cublas = B_i (N,K), B_cublas = A_i as (K,M_i) column-major (same bytes as row-major input).
//      C = A*B = weight * input^T = D_i^T.  transa=N, transb=N, ldb=K.
// Per group: m=N, n=M_i, k=K; lda=N, ldb=K, ldc=N. A/B/C arrays are device ptrs to each group's base.

class CublasGroupedKernel: public Kernel {
public:
    explicit CublasGroupedKernel(): cublas_{}
    {
        cublasCreate(&cublas_);
        cublasSetWorkspace(cublas_, nullptr, 0);
        cublasSetMathMode(cublas_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);  // match Reference::gemm

        desc_.backend    = 1;
        desc_.group_axis = 0;  // batch dim along M (ragged M per group)
        desc_.arch       = 1000;
        desc_.order_a    = kRowMajor;
        desc_.order_b    = kColMajor;
        desc_.order_c    = kRowMajor;
        desc_.type_a     = turbomind::kBfloat16;  // match MoE; Half also supported in is_feasible
        desc_.type_b     = turbomind::kBfloat16;
        desc_.type_c     = turbomind::kBfloat16;
        desc_.striding_a = Striding::kIndexed;
        desc_.striding_b = Striding::kBlocked;
        desc_.striding_c = Striding::kBlocked;
        desc_.align      = {1, 1, 1};
        desc_.cta_tile   = {256, 256, 1};  // batch_dim uses .x when group_axis=0; allow batch_size up to 256

        info_.chunk_size_k      = 1;
        info_.dynamic_smem_size = 0;
        info_.name              = GetName();
    }

    ~CublasGroupedKernel() override
    {
        cublasDestroy(cublas_);
        cublas_ = {};
    }

    int Launch(const Operation&    operation,
               float               alpha,
               const void*         A,
               const MatrixLayout& Adesc,
               const void*         U,
               const MatrixLayout& Udesc,
               const void*         B,
               const MatrixLayout& Bdesc,
               const void*         V,
               const MatrixLayout& Vdesc,
               float               beta,
               const void*         C,
               const MatrixLayout& Cdesc,
               void*               D,
               const MatrixLayout& Ddesc,
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) override
    {
        if (!Adesc.offsets || !Ddesc.offsets || Adesc.offsets == reinterpret_cast<int*>(1) || Ddesc.offsets == reinterpret_cast<int*>(1)) {
            fprintf(stderr,
                    "[TM][GEMM] CublasGrouped: missing or invalid offsets (Adesc.offsets=%p Ddesc.offsets=%p) num=%d rows=%d\n",
                    (void*)Adesc.offsets, (void*)Ddesc.offsets, Adesc.num, Adesc.rows);
            return 1;
        }
        const int group_count = Adesc.num;
        if (group_count <= 0 || Bdesc.num != group_count || Ddesc.num != group_count) {
            fprintf(stderr,
                    "[TM][GEMM] CublasGrouped: group/num mismatch group_count=%d Bdesc.num=%d Ddesc.num=%d\n",
                    group_count, Bdesc.num, Ddesc.num);
            return 1;
        }
        (void)cudaGetLastError();

        if (stream_ != stream) {
            cublasSetStream(cublas_, stream);
            stream_ = stream;
        }

        std::vector<int>         host_offsets;
        const int*              ptr_offsets = Adesc.offsets;
        cudaPointerAttributes   attr{};
        if (cudaPointerGetAttributes(&attr, (void*)Adesc.offsets) == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            host_offsets.resize(group_count + 1);
            cudaStreamSynchronize(stream);
            if (cudaMemcpy(host_offsets.data(), Adesc.offsets, (group_count + 1) * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
                fprintf(stderr, "[TM][GEMM] CublasGrouped: D2H offsets failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            ptr_offsets = host_offsets.data();
        }

        const cudaDataType cuda_type = turbomind::to_cuda_dtype(Adesc.type);
        const size_t       elem_size = turbomind::byte_size(Adesc.type, 1);

        const int N = Bdesc.cols;
        const int K = Adesc.cols;

        if (ptr_offsets[group_count] != Adesc.rows) {
            fprintf(stderr,
                    "[TM][GEMM] CublasGrouped: offsets[%d]=%d != Adesc.rows=%d (would OOB)\n",
                    group_count, ptr_offsets[group_count], Adesc.rows);
            return 1;
        }
        if (Adesc.ld < K || Ddesc.ld < N) {
            fprintf(stderr,
                    "[TM][GEMM] CublasGrouped: Adesc.ld=%d (need >= K=%d) or Ddesc.ld=%d (need >= N=%d)\n",
                    Adesc.ld, K, Ddesc.ld, N);
            return 1;
        }
        const int ldc_val = N;

        std::vector<int> m_array(group_count, N);    // cublas A rows = N (weight)
        std::vector<int> n_array(group_count);        // cublas B cols = M_i
        std::vector<int> k_array(group_count, K);
        std::vector<int> lda_array(group_count, N);   // weight (N,K), lda=N
        std::vector<int> ldb_array(group_count, K);   // input^T (K,M_i), ldb=K
        std::vector<int> ldc_array(group_count, ldc_val);  // C (N,M_i), ldc=N (match Reference::gemm)

        std::vector<const void*> a_ptrs(group_count);  // A = weight
        std::vector<const void*> b_ptrs(group_count);  // B = input tile
        std::vector<void*>       c_ptrs(group_count);

        const bool weight_is_strided_ptrs = (Bdesc.ld == 0);
        const uintptr_t kBadB = 0x320936400ULL;
        if (weight_is_strided_ptrs && (B == nullptr || reinterpret_cast<uintptr_t>(B) == kBadB)) {
            fprintf(stderr, "[TM][GEMM] CublasGrouped: B null or bad (B=%p)\n", (void*)B);
            return 1;
        }

        for (int i = 0; i < group_count; ++i) {
            const int M_i    = ptr_offsets[i + 1] - ptr_offsets[i];
            n_array[i]       = M_i;
            const int off_a  = ptr_offsets[i] * Adesc.ld;
            const int off_d  = ptr_offsets[i] * Ddesc.ld;

            if (!weight_is_strided_ptrs) {
                const int off_b = Bdesc.offsets ? Bdesc.offsets[i] * Bdesc.ld : i * (K * N);
                a_ptrs[i] = static_cast<const char*>(B) + off_b * elem_size;
            }
            b_ptrs[i] = static_cast<const char*>(A) + off_a * elem_size;
            c_ptrs[i] = static_cast<char*>(D) + off_d * elem_size;
        }

        std::vector<int> active_idx;
        active_idx.reserve(group_count);
        for (int i = 0; i < group_count; ++i) {
            if (n_array[i] > 0)
                active_idx.push_back(i);
        }
        const int active_count = (int)active_idx.size();
        if (active_count == 0) {
            return 0;  // no non-empty groups
        }

        std::vector<int>         m_active(active_count), n_active(active_count), k_active(active_count);
        std::vector<int>         lda_active(active_count), ldb_active(active_count), ldc_active(active_count);
        std::vector<const void*> a_ptrs_active(active_count);
        std::vector<const void*> b_ptrs_active(active_count);
        std::vector<void*>       c_ptrs_active(active_count);
        for (int j = 0; j < active_count; ++j) {
            int i            = active_idx[j];
            m_active[j]      = m_array[i];
            n_active[j]      = n_array[i];
            k_active[j]      = k_array[i];
            lda_active[j]    = lda_array[i];
            ldb_active[j]    = ldb_array[i];
            ldc_active[j]    = ldc_array[i];
            a_ptrs_active[j] = a_ptrs[i];
            b_ptrs_active[j] = b_ptrs[i];
            c_ptrs_active[j] = c_ptrs[i];
        }

        // Same as loop path: transa=N, transb=N; row-major input (M_i,K) read as (K,M_i) col-major, ldb=K.
        std::vector<cublasOperation_t> transa_array(active_count, CUBLAS_OP_N);
        std::vector<cublasOperation_t> transb_array(active_count, CUBLAS_OP_N);
        std::vector<float>             alpha_array(active_count, alpha);
        std::vector<float>             beta_array(active_count, beta);
        std::vector<int>               group_size(active_count, 1);

        const size_t ptr_buf_size = active_count * sizeof(void*);
        void*        d_a_ptrs     = nullptr;
        void*        d_b_ptrs     = nullptr;
        void*        d_c_ptrs     = nullptr;
        if (cudaMallocAsync(&d_a_ptrs, ptr_buf_size, stream) != cudaSuccess ||
            cudaMallocAsync(&d_b_ptrs, ptr_buf_size, stream) != cudaSuccess ||
            cudaMallocAsync(&d_c_ptrs, ptr_buf_size, stream) != cudaSuccess) {
            fprintf(stderr, "[TM][GEMM] CublasGrouped: cudaMallocAsync ptr arrays failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return 1;
        }

        if (weight_is_strided_ptrs) {
            // B = dense.weight.raw_data(): device pointer to StridedPtr[group_count]. a_ptrs[i] = expert weight base.
            cudaPointerAttributes attr{};
            if (cudaPointerGetAttributes(&attr, B) != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
                fprintf(stderr, "[TM][GEMM] CublasGrouped: B not device ptr (attr.type=%d)\n", (int)attr.type);
                cudaFreeAsync(d_a_ptrs, stream);
                cudaFreeAsync(d_b_ptrs, stream);
                cudaFreeAsync(d_c_ptrs, stream);
                return 1;
            }
            cudaStreamSynchronize(stream);
            std::vector<StridedPtr> host_strided(group_count);
            cudaError_t err = cudaMemcpy(host_strided.data(), B, group_count * sizeof(StridedPtr), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "[TM][GEMM] CublasGrouped: D2H B (StridedPtr) failed: %s\n", cudaGetErrorString(err));
                cudaFreeAsync(d_a_ptrs, stream);
                cudaFreeAsync(d_b_ptrs, stream);
                cudaFreeAsync(d_c_ptrs, stream);
                return 1;
            }
            const void* const kBadAddr = reinterpret_cast<void*>(0x320936400ULL);
            for (int i = 0; i < group_count; ++i) {
                a_ptrs[i] = host_strided[i].ptr;
                lda_array[i] = N;
                if (!a_ptrs[i] || a_ptrs[i] == kBadAddr) {
                    fprintf(stderr, "[TM][GEMM] CublasGrouped: weight ptr[%d]=%p (null or sentinel)\n", i, (void*)a_ptrs[i]);
                    cudaFreeAsync(d_a_ptrs, stream);
                    cudaFreeAsync(d_b_ptrs, stream);
                    cudaFreeAsync(d_c_ptrs, stream);
                    return 1;
                }
            }
            for (int j = 0; j < active_count; ++j) {
                lda_active[j]    = lda_array[active_idx[j]];
                a_ptrs_active[j] = a_ptrs[active_idx[j]];
            }
        }
        cudaMemcpyAsync(d_a_ptrs, a_ptrs_active.data(), ptr_buf_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b_ptrs, b_ptrs_active.data(), ptr_buf_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_c_ptrs, c_ptrs_active.data(), ptr_buf_size, cudaMemcpyHostToDevice, stream);

        // Sync so device sees pointer arrays before cuBLAS (avoids EXECUTION_FAILED on SM100).
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            fprintf(stderr, "[TM][GEMM] CublasGrouped: sync before GEMM failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            cudaFreeAsync(d_a_ptrs, stream);
            cudaFreeAsync(d_b_ptrs, stream);
            cudaFreeAsync(d_c_ptrs, stream);
            return 1;
        }

        cublasStatus_t status = cublasGemmGroupedBatchedEx(cublas_,
                                                           transa_array.data(),
                                                           transb_array.data(),
                                                           m_active.data(),
                                                           n_active.data(),
                                                           k_active.data(),
                                                           alpha_array.data(),
                                                           reinterpret_cast<const void* const*>(d_a_ptrs),
                                                           cuda_type,
                                                           lda_active.data(),
                                                           reinterpret_cast<const void* const*>(d_b_ptrs),
                                                           cuda_type,
                                                           ldb_active.data(),
                                                           beta_array.data(),
                                                           reinterpret_cast<void* const*>(d_c_ptrs),
                                                           cuda_type,
                                                           ldc_active.data(),
                                                           active_count,
                                                           group_size.data(),
                                                           CUBLAS_COMPUTE_32F);

        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[TM][GEMM] CublasGrouped: cublasGemmGroupedBatchedEx failed: %s\n", _cudaGetErrorEnum(status));
            cudaFreeAsync(d_a_ptrs, stream);
            cudaFreeAsync(d_b_ptrs, stream);
            cudaFreeAsync(d_c_ptrs, stream);
            return 1;
        }
        // Sync before free so cublas (and any internal kernels) finish using the pointer arrays
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        cudaFreeAsync(d_a_ptrs, stream);
        cudaFreeAsync(d_b_ptrs, stream);
        cudaFreeAsync(d_c_ptrs, stream);
        if (sync_err != cudaSuccess) {
            fprintf(stderr, "[TM][GEMM] CublasGrouped: cudaStreamSynchronize failed: %s\n", cudaGetErrorString(sync_err));
            return 1;
        }
        return 0;
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        if (desc.num <= 1 || desc.group_axis < 0) {
            return false;
        }
        // Reject group_axis=1 (transposed): TransposedKernel swaps A and B, so CublasGroupedKernel would receive
        // weight descriptor as Adesc; weight has no valid offsets -> Adesc.offsets=(nil) and Launch fails.
        if (desc.group_axis != 0) {
            return false;
        }
        if (!is_arch_compatible(desc_.arch, desc.arch)) {
            return false;
        }
        if (desc.striding_a != Striding::kBlocked && desc.striding_a != Striding::kIndexed) {
            return false;
        }
        if (desc.striding_c != Striding::kBlocked && desc.striding_c != Striding::kIndexed) {
            return false;
        }
        if (desc.striding_b != Striding::kFlat && desc.striding_b != Striding::kBlocked) {
            return false;
        }
        // Allow any epilogue; Launch does plain GEMM, caller applies epilogue if needed
        if (desc.quant_a || desc.quant_b) {
            return false;
        }
        if (desc.type_a != kHalf && desc.type_a != kBfloat16) {
            return false;
        }
        if (desc.type_b != desc.type_a || desc.type_c != desc.type_a) {
            return false;
        }
        return true;
    }

    int GetMaxSwizzle(const int4&) const override { return 0; }
    int GetMaxSplits(const int4&, int, size_t, size_t) const override { return 1; }

private:
    cublasHandle_t cublas_{};
    cudaStream_t   stream_{};
};

void Registry::sm100_cublas_grouped_float()
{
    Add(std::make_unique<CublasGroupedKernel>());
}

#endif  // ENABLE_CUBLAS_GROUPED

}  // namespace turbomind::gemm
