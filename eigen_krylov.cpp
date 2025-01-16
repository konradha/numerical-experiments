// clang++ -std=c++20 eigen_krylov.cpp -I$CONDA_PREFIX/include/eigen3
// -Wl,-rpath,$CONDA_PREFIX/lib -o eigen-solver

#include "npy.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <string>
#include <vector>

#include <chrono>
#include <iomanip>
#include <utility>

template <typename Float>
Eigen::SparseMatrix<double> buildD2(uint32_t nx, uint32_t ny, Float dx,
                                    Float dy) {
  assert(nx == ny);
  assert(std::abs(dx - dy) < 1e-10);

  const uint32_t N = (nx + 2) * (nx + 2);
  const uint32_t nnz = N + 4 * (N - 1) - 4;
  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets;
  triplets.reserve(nnz);

  for (uint32_t i = 0; i < N; ++i) {
    Float val = static_cast<Float>(-4.0);
    if (i < (nx + 2) || i >= N - (nx + 2) || i % (nx + 2) == 0 ||
        i % (nx + 2) == (nx + 1)) {
      val = static_cast<Float>(-3.0);
    }
    triplets.emplace_back(i, i, val);
  }

  for (uint32_t i = 0; i < N - 1; ++i) {
    if ((i + 1) % (nx + 2) != 0) {
      triplets.emplace_back(i, i + 1, static_cast<Float>(1.0));
      triplets.emplace_back(i + 1, i, static_cast<Float>(1.0));
    }
  }

  for (uint32_t i = 0; i < N - (nx + 2); ++i) {
    triplets.emplace_back(i, i + (nx + 2), static_cast<Float>(1.0));
    triplets.emplace_back(i + (nx + 2), i, static_cast<Float>(1.0));
  }

  Eigen::SparseMatrix<Float> L(N, N);
  L.setFromTriplets(triplets.begin(), triplets.end());
  L.makeCompressed();
  L *= static_cast<Float>(1.0) / (dx * dy);
  return L;
}

template <typename Float, typename F>
Eigen::VectorX<Float> apply_function(const Eigen::VectorX<Float> &x,
                                     const Eigen::VectorX<Float> &y, F f) {
  const uint32_t nx = x.size();
  const uint32_t ny = y.size();
  Eigen::VectorX<Float> u(nx * ny);
  for (uint32_t i = 0; i < ny; ++i) {
    for (uint32_t j = 0; j < nx; ++j) {
      u[i * nx + j] = f(x[i], y[j]);
    }
  }
  return u;
}

template <typename Float, typename F>
Eigen::VectorX<Float> apply_function_uniform(Float x_min, Float x_max,
                                             uint32_t nx, Float y_min,
                                             Float y_max, uint32_t ny, F f) {
  Eigen::VectorX<Float> x = Eigen::VectorX<Float>::LinSpaced(nx, x_min, x_max);
  Eigen::VectorX<Float> y = Eigen::VectorX<Float>::LinSpaced(ny, y_min, y_max);
  return apply_function<Float>(x, y, f);
}

template <typename Float>
void apply_neumann_bc(Eigen::VectorX<Float> &u, Eigen::VectorX<Float> &v,
                      const uint32_t nx, const uint32_t ny) {
  // no copy, just another view
  Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> u_2d(
      u.data(), nx, ny);
  Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> v_2d(
      v.data(), nx, ny);

  u_2d.row(0).segment(1, ny - 2) = u_2d.row(1).segment(1, ny - 2);
  u_2d.row(nx - 1).segment(1, ny - 2) = u_2d.row(nx - 2).segment(1, ny - 2);
  u_2d.col(0) = u_2d.col(1);
  u_2d.col(ny - 1) = u_2d.col(ny - 2);

  v_2d.row(0).segment(1, ny - 2).setZero();
  v_2d.row(nx - 1).segment(1, ny - 2).setZero();
  v_2d.col(0).segment(1, nx - 2).setZero();
  v_2d.col(ny - 1).segment(1, nx - 2).setZero();
}

template <typename Float>
void save_to_npy(const std::string &filename, const Eigen::VectorX<Float> &data,
                 const std::vector<uint32_t> &shape) {
  std::vector<Float> vec(data.data(), data.data() + data.size());
  std::vector<uint64_t> shape_ul;
  for (const auto dim : shape)
    shape_ul.push_back(static_cast<uint64_t>(dim));
  for (const auto dim : shape)
    std::cout << dim << " ";
  std::cout << "\n"
            << "shape size: " << shape.size() << "\n";

  npy::SaveArrayAsNumpy(filename, false, shape.size(), shape_ul.data(), vec);
}

template <typename Float>
Eigen::VectorXd laplacian(const Eigen::VectorX<Float> &u, const uint32_t nx,
                          const uint32_t ny, Float dx) {
  const Float dx2_inv = 1.0 / (dx * dx);
  Eigen::VectorX<Float> result(nx * ny);
  const uint32_t up = -ny;
  const uint32_t down = ny;
  const uint32_t left = -1;
  const uint32_t right = 1;
  constexpr uint32_t BLOCK_SIZE = 64;// / sizeof(double);
  for (uint32_t ib = 1; ib < nx - 1; ib += BLOCK_SIZE) {
    for (uint32_t jb = 1; jb < ny - 1; jb += BLOCK_SIZE) {
      const uint32_t imax = std::min(ib + BLOCK_SIZE, nx - 1);
      const uint32_t jmax = std::min(jb + BLOCK_SIZE, ny - 1);

      for (uint32_t i = ib; i < imax; ++i) {
        const uint32_t base_idx = i * ny + jb;
        _mm_prefetch(&u[base_idx + down], _MM_HINT_T0);
        for (uint32_t j = 0; j < jmax - jb; ++j) {
          const uint32_t idx = base_idx + j;
          result[idx] = (u[idx + up] + u[idx + down] + u[idx + left] +
                         u[idx + right] - 4.0 * u[idx]) *
                        dx2_inv;
        }
      }
    }
  }
  return result;
}

struct BenchmarkResult {
  double sparse_time;
  double stencil_time;
  size_t size;
  size_t iterations;
  double l1_norm_diff;
  double l2_norm_diff;
};

std::vector<BenchmarkResult>
benchmark_laplacian(const std::vector<uint32_t> &sizes,
                    size_t iterations = 100) {
  std::vector<BenchmarkResult> results;
  results.reserve(sizes.size());

  for (const auto n : sizes) {
    const uint32_t nx = n;
    const uint32_t ny = n;
    const double L = 3.0;
    const double dx = 2.0 * L / nx;
    const double dy = 2.0 * L / ny;
    auto gaussian = [](double x, double y) {
      return std::exp(-(x * x + y * y));
    };
    auto u = apply_function_uniform<double>(-L, L, nx, -L, L, ny, gaussian);
    const auto Laplacian = buildD2<double>(nx - 2, ny - 2, dx, dy);

    Eigen::VectorXd sparse_result = Laplacian * u;
    Eigen::VectorXd stencil_result = laplacian(u, nx, ny, dx);

    Eigen::Map<Eigen::MatrixXd> sparse_2d(sparse_result.data(), nx, ny);
    Eigen::Map<Eigen::MatrixXd> stencil_2d(stencil_result.data(), nx, ny);

    Eigen::MatrixXd sparse_interior = sparse_2d.block(1, 1, nx-2, ny-2);
    Eigen::MatrixXd stencil_interior = stencil_2d.block(1, 1, nx-2, ny-2);

    double l1_norm_diff = (sparse_interior - stencil_interior).lpNorm<1>();
    double l2_norm_diff = (sparse_interior - stencil_interior).lpNorm<2>();


    volatile auto warmup_sparse = Laplacian * u;
    volatile auto warmup_stencil = laplacian(u, nx, ny, dx);

    auto start = std::chrono::high_resolution_clock::now();
    double check_mv = 0.;
    for (size_t i = 0; i < iterations; ++i) {
      auto result = Laplacian * u;
      auto s = result.sum();
      check_mv += s;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double sparse_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    double check_stencil = 0.;
    for (size_t i = 0; i < iterations; ++i) {
      auto result = laplacian(u, nx, ny, dx);
      auto s = result.sum();
      check_stencil += s;
    }
    end = std::chrono::high_resolution_clock::now();
    double stencil_time = std::chrono::duration<double>(end - start).count();
    results.push_back({sparse_time/iterations, stencil_time/iterations, n, iterations, l1_norm_diff, l2_norm_diff});
  }
  return results;
}

void print_benchmark_results(const std::vector<BenchmarkResult> &results) {
  std::cout << std::setw(10) << "Size" << std::setw(15) << "Sparse (s)"
            << std::setw(15) << "Stencil (s)" << std::setw(15) << "Ratio"
            << std::setw(15) << "Iterations"
            << std::setw(20) << "L1 Norm Diff"
            << std::setw(20) << "L2 Norm Diff" << "\n";
            
  std::cout << std::string(110, '-') << "\n";

  for (const auto &result : results) {
    std::cout << std::setw(10) << result.size << std::setw(15) << std::fixed
              << std::setprecision(6) << result.sparse_time << std::setw(15)
              << result.stencil_time << std::setw(15)
              << result.sparse_time / result.stencil_time << std::setw(15)
              << result.iterations << std::setw(20)
              << result.l1_norm_diff << std::setw(20)
              << result.l2_norm_diff << "\n";
  }
}

void benchmark() {
  std::vector<uint32_t> sizes = {64, 128, 256, 512, 1024};
  const size_t iterations = 20;
  auto results = benchmark_laplacian(sizes, iterations);
  print_benchmark_results(results);
}

int main() {
  benchmark();
  /*
  // TODO capture n = nx = ny, L via argv
  uint32_t nx = 100;
  uint32_t ny = 100;

  double L = 3;
  double dx = 2 * L / nx;
  double dy = 2 * L / ny;

  const auto Laplacian = buildD2<double>(nx - 2, ny - 2, dx, dy);

  auto gaussian = [](double x, double y) { return std::exp(-(x * x + y * y)); };
  auto gaussian_del = [&](double x, double y) {
      return (-4 + 4. * (x * x + y * y)) * gaussian(x, y);};
  const std::vector<uint32_t> shape = {nx, ny};
  auto u = apply_function_uniform(-L, L, nx, -L, L, ny, gaussian);

  Eigen::VectorX<double> del_u = Laplacian * u;
  //Eigen::VectorX<double> del_u_analytical = apply_function_uniform(-L, L, nx,
  -L, L, ny, gaussian_del); Eigen::VectorX<double> del_u_stencil = laplacian(u,
  nx, ny, dx);
  //Eigen::VectorX<double> diff = del_u - del_u_analytical;



  const auto fname = "/home/konrad/code/msc-thesis/from-scratch/lapl_mat.npy";
  save_to_npy(fname, del_u, shape);
  const auto fname_del =
      "/home/konrad/code/msc-thesis/from-scratch/lapl_stencil.npy";
  save_to_npy(fname_del, del_u_stencil, shape);
  */
}
