#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <Eigen/Sparse>
#include <memory>

template<typename Float>
Eigen::SparseMatrix<Float> constructLaplacian2D(
   uint32_t nx, uint32_t ny, Float dx, Float dy) {
   const uint32_t n = nx * ny;
   std::vector<Eigen::Triplet<Float>> triplets;
   triplets.reserve(5 * n);
   
   for(uint32_t i = 0; i < ny; ++i) {
       for(uint32_t j = 0; j < nx; ++j) {
           uint32_t idx = i * ny + j;
           triplets.emplace_back(idx, idx, -2.0/dx/dx - 2.0/dy/dy);
           
           if(i > 0) triplets.emplace_back(idx, (i-1)*ny + j, 1.0/dx/dx);
           if(i < nx-1) triplets.emplace_back(idx, (i+1)*ny + j, 1.0/dx/dx);
           if(j > 0) triplets.emplace_back(idx, i*ny + (j-1), 1.0/dy/dy);
           if(j < ny-1) triplets.emplace_back(idx, i*ny + (j+1), 1.0/dy/dy);
       }
   }
   
   Eigen::SparseMatrix<Float> L(n, n);
   L.setFromTriplets(triplets.begin(), triplets.end());
   return L;
}

template<typename Float>
struct CPUContext {
   using Vector = Eigen::VectorX<Float>;
   using Matrix = Eigen::SparseMatrix<Float>;

   void gradH_q(const Vector& u, Vector& result,
               const Matrix& L, const Vector& mass,
               const Vector& c) {
       result = c.array() * (L * u).array() - mass.array() * u.array().sin();
   }

   void stormer_verlet_step(Vector& unext, Vector& vnext,
                           const Vector& u, const Vector& upast,
                           const Vector& gradient, Float tau) {
       unext = 2.0 * u - upast + 0.5 * tau * tau * gradient;
       vnext = (unext - upast) / (2.0 * tau);
   }
};

template<typename Float>
struct GPUContext {
   using Vector = thrust::device_vector<Float>;
   using Matrix = cusparseSpMatDescr_t;

   cusparseHandle_t sparse_handle;
   void* buffer;

   GPUContext() : buffer(nullptr) {
       cusparseCreate(&sparse_handle);
   }

   void spmv(const Matrix& A, const Vector& x, Vector& y) {
       const Float alpha = 1.0, beta = 0.0;
       const void* alpha_ptr = &alpha;
       const void* beta_ptr = &beta;
       
       cusparseDnVecDescr_t vecX, vecY;
       cusparseCreateDnVec(&vecX, x.size(), 
           const_cast<void*>(static_cast<const void*>(thrust::raw_pointer_cast(x.data()))), 
           CUDA_R_64F);
       cusparseCreateDnVec(&vecY, y.size(), 
           thrust::raw_pointer_cast(y.data()), 
           CUDA_R_64F);
       
       cusparseSpMV(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
           alpha_ptr, A, vecX, beta_ptr, vecY, CUDA_R_64F,
           CUSPARSE_SPMV_ALG_DEFAULT, buffer);

       cusparseDestroyDnVec(vecX);
       cusparseDestroyDnVec(vecY);
   }

   struct gradH_functor {
       const Float* mass;
       const Float* c;
       const Float* lap;
       
       gradH_functor(const Float* m, const Float* c_, const Float* l) 
           : mass(m), c(c_), lap(l) {}
       
       __host__ __device__
       Float operator()(const thrust::tuple<Float, uint32_t>& t) const {
           const uint32_t i = thrust::get<1>(t);
           return c[i] * lap[i] - mass[i] * sin(thrust::get<0>(t));
       }
   };

   void gradH_q(const Vector& u, Vector& result,
                const Matrix& L, const Vector& mass,
                const Vector& c) {
       Vector lap(u.size());
       spmv(L, u, lap);
       
       thrust::transform(thrust::device,
           thrust::make_zip_iterator(thrust::make_tuple(
               u.begin(), thrust::make_counting_iterator<uint32_t>(0))),
           thrust::make_zip_iterator(thrust::make_tuple(
               u.end(), thrust::make_counting_iterator<uint32_t>(u.size()))),
           result.begin(),
           gradH_functor(
               thrust::raw_pointer_cast(mass.data()),
               thrust::raw_pointer_cast(c.data()),
               thrust::raw_pointer_cast(lap.data())
           )
       );
   }

   void stormer_verlet_step(Vector& unext, Vector& vnext,
                           const Vector& u, const Vector& upast,
                           const Vector& gradient, Float tau) {
       thrust::transform(thrust::device,
           thrust::make_zip_iterator(thrust::make_tuple(
               u.begin(), upast.begin(), gradient.begin())),
           thrust::make_zip_iterator(thrust::make_tuple(
               u.end(), upast.end(), gradient.end())),
           unext.begin(),
           [tau] __device__ (const thrust::tuple<Float,Float,Float>& t) {
               return 2.0f * thrust::get<0>(t) - thrust::get<1>(t) + 
                      0.5f * tau * tau * thrust::get<2>(t);
           }
       );

       thrust::transform(thrust::device,
           thrust::make_zip_iterator(thrust::make_tuple(unext.begin(), upast.begin())),
           thrust::make_zip_iterator(thrust::make_tuple(unext.end(), upast.end())),
           vnext.begin(),
           [tau] __device__ (const thrust::tuple<Float,Float>& t) {
               return (thrust::get<0>(t) - thrust::get<1>(t)) / (2.0f * tau);
           }
       );
   }

   ~GPUContext() {
       if (buffer) cudaFree(buffer);
       cusparseDestroy(sparse_handle);
   }
};

template<typename Float, typename Context>
class SineGordonSolver {
   using Vector = typename Context::Vector;
   using Matrix = typename Context::Matrix;

   const uint32_t nx, ny, n;
   const Float dx, dy;
   std::unique_ptr<Context> context;
   Vector u, v, upast, ubuf, mass, c;
   Matrix laplacian;

public:
   SineGordonSolver(uint32_t nx_, uint32_t ny_, Float dx_, Float dy_,
                    const Eigen::SparseMatrix<Float>& L,
                    const Eigen::VectorX<Float>& m,
                    const Eigen::VectorX<Float>& c_)
       : nx(nx_), ny(ny_), n(nx * ny),
         dx(dx_), dy(dy_),
         context(std::make_unique<Context>()) {
       
       if constexpr (std::is_same_v<Context, CPUContext<Float>>) {
           laplacian = L;
           mass = m;
           c = c_;
       } else {
           cusparseCreateCsr(&laplacian, n, n, L.nonZeros(),
               const_cast<int*>(L.outerIndexPtr()),
               const_cast<int*>(L.innerIndexPtr()),
               const_cast<Float*>(L.valuePtr()),
               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

           const Float alpha = 1.0, beta = 0.0;
           const void* alpha_ptr = &alpha;
           const void* beta_ptr = &beta;
           
           size_t buffer_size;
           cusparseSpMV_bufferSize(
               context->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               alpha_ptr, laplacian, nullptr, beta_ptr, nullptr,
               CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
           
           cudaMalloc(&context->buffer, buffer_size);
           
           mass = Vector(m.data(), m.data() + n);
           c = Vector(c_.data(), c_.data() + n);
       }
       
       u.resize(n);
       v.resize(n);
       upast.resize(n);
       ubuf.resize(n);
   }

   void setState(const Eigen::VectorX<Float>& u0,
                const Eigen::VectorX<Float>& v0,
                Float tau) {
       if constexpr (std::is_same_v<Context, CPUContext<Float>>) {
           u = u0;
           v = v0;
           upast = u0 - tau * v0;
       } else {
           thrust::copy(u0.data(), u0.data() + n, u.begin());
           thrust::copy(v0.data(), v0.data() + n, v.begin());
           thrust::transform(thrust::device,
               thrust::make_zip_iterator(thrust::make_tuple(u.begin(), v.begin())),
               thrust::make_zip_iterator(thrust::make_tuple(u.end(), v.end())),
               upast.begin(),
               [tau] __device__ (const thrust::tuple<Float,Float>& t) {
                   return thrust::get<0>(t) - tau * thrust::get<1>(t);
               }
           );
       }
   }

   void step(Float tau) {
       context->gradH_q(u, ubuf, laplacian, mass, c);
       context->stormer_verlet_step(u, v, u, upast, ubuf, tau);
       std::swap(u, upast);
   }

   void getState(Eigen::VectorX<Float>& u_out, Eigen::VectorX<Float>& v_out) {
       u_out.resize(n);
       v_out.resize(n);
       if constexpr (std::is_same_v<Context, CPUContext<Float>>) {
           u_out = u;
           v_out = v;
       } else {
           thrust::copy(u.begin(), u.end(), u_out.data());
           thrust::copy(v.begin(), v.end(), v_out.data());
           cudaDeviceSynchronize();
       }
   }

   ~SineGordonSolver() {
       if constexpr (!std::is_same_v<Context, CPUContext<Float>>) {
           cusparseDestroySpMat(laplacian);
       }
   }
};

using SineGordonCPU = SineGordonSolver<double, CPUContext<double>>;
using SineGordonGPU = SineGordonSolver<double, GPUContext<double>>;
