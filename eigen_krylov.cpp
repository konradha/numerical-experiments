// clang++ -std=c++20 -march=native -O2 eigen_krylov.cpp
// -I$CONDA_PREFIX/include/eigen3 -Wl,-rpath,$CONDA_PREFIX/lib -o eigen-solver
#include "npy.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define DEBUG 0

#define PROGRESS_BAR(i, total) \
    if ((i + 1) % (total/100 + 1) == 0 || i + 1 == total) { \
        float progress = (float)(i + 1)/total; \
        int barWidth = 70; \
        std::cout << "["; \
        int pos = barWidth * progress; \
        for (int i = 0; i < barWidth; ++i) { \
            if (i < pos) std::cout << "="; \
            else if (i == pos) std::cout << ">"; \
            else std::cout << " "; \
        } \
        std::cout << "] " << int(progress * 100.0) << "% \r"; \
        std::cout.flush(); \
        if (i + 1 == total) std::cout << std::endl; \
    }

template <typename F, typename Scalar>
concept ValidScalarLambda =
    !std::is_same_v<std::decay_t<F>, std::function<Scalar(Scalar)>> &&
    std::is_invocable_r_v<Scalar, F, Scalar>;

template <typename StepFn, typename Float, typename NonlinearF>
concept StepFunction = requires(
    StepFn fn, Eigen::VectorX<Float> &unext, Eigen::VectorX<Float> &vnext,
    const Eigen::VectorX<Float> &un, const Eigen::VectorX<Float> &vn,
    const Eigen::VectorX<Float> &upast, Eigen::VectorX<Float> &ubuf,
    const Eigen::SparseMatrix<Float> &Laplacian, uint32_t nx, uint32_t ny,
    Float dx, Float dy, Float tau, NonlinearF nonlinear_term) {
  {
    fn(unext, vnext, un, vn, upast, ubuf, Laplacian, nx, ny, dx, dy, tau,
       nonlinear_term)
    } -> std::same_as<void>;
  requires std::is_invocable_r_v<Float, NonlinearF, Float>;
};

template <typename Float>
Eigen::SparseMatrix<Float> buildD2(uint32_t nx, uint32_t ny, Float dx,
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
  constexpr uint32_t BLOCK_SIZE = 64; // / sizeof(double);
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

    Eigen::MatrixXd sparse_interior = sparse_2d.block(1, 1, nx - 2, ny - 2);
    Eigen::MatrixXd stencil_interior = stencil_2d.block(1, 1, nx - 2, ny - 2);

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
    results.push_back({sparse_time / iterations, stencil_time / iterations, n,
                       iterations, l1_norm_diff, l2_norm_diff});
  }
  return results;
}

void print_benchmark_results(const std::vector<BenchmarkResult> &results) {
  std::cout << std::setw(10) << "Size" << std::setw(15) << "Sparse (s)"
            << std::setw(15) << "Stencil (s)" << std::setw(15) << "Ratio"
            << std::setw(15) << "Iterations" << std::setw(20) << "L1 Norm Diff"
            << std::setw(20) << "L2 Norm Diff"
            << "\n";

  std::cout << std::string(110, '-') << "\n";

  for (const auto &result : results) {
    std::cout << std::setw(10) << result.size << std::setw(15) << std::fixed
              << std::setprecision(6) << result.sparse_time << std::setw(15)
              << result.stencil_time << std::setw(15)
              << result.sparse_time / result.stencil_time << std::setw(15)
              << result.iterations << std::setw(20) << result.l1_norm_diff
              << std::setw(20) << result.l2_norm_diff << "\n";
  }
}

void benchmark() {
  std::vector<uint32_t> sizes = {64, 128, 256, 512, 1024};
  const size_t iterations = 20;
  auto results = benchmark_laplacian(sizes, iterations);
  print_benchmark_results(results);
}

template <typename Float, typename F>
requires ValidScalarLambda<F, Float>
void gradH_q_stencil(const Eigen::VectorX<Float> &un,
                     Eigen::VectorX<Float> &ubuf, const uint32_t nx,
                     const uint32_t ny, const Float dx, const Float dy, F fun) {
  ubuf = un.array().unaryExpr(fun);
  ubuf = ubuf + laplacian(un, nx, ny, dx);
}

template <typename Float, typename F>
requires ValidScalarLambda<F, Float>
void gradH_q_spmv(const Eigen::VectorX<Float> &un, Eigen::VectorX<Float> &ubuf,
                  const Eigen::SparseMatrix<Float> &Laplacian,
                  const uint32_t nx, const uint32_t ny, const Float dx,
                  const Float dy, F fun) {

  ubuf = un.array().unaryExpr(fun);
  ubuf = ubuf + Laplacian * un;
}

template <typename Float, typename F>
void stormer_verlet_step_stencil(
    Eigen::VectorX<Float> &unext, Eigen::VectorX<Float> &vnext,
    const Eigen::VectorX<Float> &un, const Eigen::VectorX<Float> &vn,
    const Eigen::VectorX<Float> &upast, Eigen::VectorX<Float> &ubuf,
    const Eigen::SparseMatrix<Float> &Laplacian, const uint32_t nx,
    const uint32_t ny, const Float dx, const Float dy, const Float tau, F fun) {
  gradH_q_stencil(un, ubuf, nx, ny, dx, dy, fun);
  unext = 2 * un - upast + .5 * tau * tau * ubuf;
  vnext = (unext - upast) * .5 / tau;
}

template <typename Float, typename F>
void stormer_verlet_step_spmv(
    Eigen::VectorX<Float> &unext, Eigen::VectorX<Float> &vnext,
    const Eigen::VectorX<Float> &un, const Eigen::VectorX<Float> &vn,
    const Eigen::VectorX<Float> &upast, Eigen::VectorX<Float> &ubuf,
    const Eigen::SparseMatrix<Float> &Laplacian, const uint32_t nx,
    const uint32_t ny, const Float dx, const Float dy, const Float tau, F fun) {
  const Eigen::SparseMatrix<Float> NLaplacian = -Laplacian; 
  gradH_q_spmv(un, ubuf, NLaplacian, nx, ny, dx, dy, fun);
  unext = 2 * un - upast + .5 * tau * tau * ubuf;
  vnext = (unext - upast) * .5 / tau;
}

template <typename Float>
std::tuple<Eigen::MatrixX<Float>, Eigen::MatrixX<Float>, Float>
lanczos_L(const Eigen::SparseMatrix<Float> &L, const Eigen::VectorX<Float> &u,
          const uint32_t m) {
  const uint32_t n = L.rows();
  Eigen::MatrixX<Float> V = Eigen::MatrixX<Float>::Zero(n, m);
  Eigen::MatrixX<Float> T = Eigen::MatrixX<Float>::Zero(m, m);

  Float beta = u.norm();
  V.col(0) = u / beta; 

  for (uint32_t j = 0; j < m - 1; j++) {
    Eigen::VectorX<Float> w = L * V.col(j);
    if (j > 0)
      w -= T(j - 1, j) * V.col(j - 1);
     
    T(j, j) = w.dot(V.col(j));
    w -= T(j, j) * V.col(j);

    for (uint32_t i = 0; i <= j; i++) {
        Float coeff = w.dot(V.col(i));
        w -= coeff * V.col(i);
    }
    T(j + 1, j) = w.norm();
#if DEBUG
    std::cout << "j=" << j << " alpha=" << T(j,j)
              << " beta=" << T(j+1,j) << "\n";
#endif
    // use symmetry
    T(j, j + 1) = T(j + 1, j);
    if (T(j + 1, j) < 1e-8) {
      V.conservativeResize(Eigen::NoChange, j + 1);
      T.conservativeResize(j + 1, j + 1);
      break;
    }
    V.col(j + 1) = w / T(j + 1, j);
  }
  return {V, T, beta};
}

template <typename Float>
Eigen::VectorX<Float> cos_L(const Eigen::SparseMatrix<Float> &L,
                            const Eigen::VectorX<Float> &u, Float t,
                            const uint32_t m = 30) {
  const auto [V, T, beta] = lanczos_L(L, u, m);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  Eigen::MatrixX<Float> cos_sqrt_T =
      (es.eigenvectors() * (t * es.eigenvalues().array().abs().sqrt()).cos().matrix().asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * cos_sqrt_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> sinc_L(const Eigen::SparseMatrix<Float> &L,
                             const Eigen::VectorX<Float> &u, Float t,
                             const uint32_t m = 30) {
  const auto [V, T, beta] = lanczos_L(L, u, m);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  
  Eigen::MatrixX<Float> sinc_sqrt_T =
       (es.eigenvectors() * (t * es.eigenvalues().array().abs().sqrt()).sinc().matrix().asDiagonal() *
       es.eigenvectors().transpose());
 

  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * sinc_sqrt_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> sinc2_L_half(const Eigen::SparseMatrix<Float> &L,
                                   const Eigen::VectorX<Float> &u, Float t,
                                   const uint32_t m = 30) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  Eigen::MatrixX<Float> sinc_sqrt_T =
      (es.eigenvectors() * (t/2. * es.eigenvalues().array().abs().sqrt()).
       unaryExpr([](Float x) {
           return std::abs(x) < 1e-8 ? Float(1) : std::sin(x)/x;
       }).square().matrix().asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * sinc_sqrt_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> Omega_scaled(const Eigen::SparseMatrix<Float> &L,
                                   const Eigen::VectorX<Float> &u, Float t,
                                   const uint32_t m = 30) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
#if DEBUG
  std::cout << "Omega_scaled\n";
  std::cout << "Eigenvalues: " << es.eigenvalues().transpose() << "\n";
  std::cout << "Beta: " << beta << "\n";
  std::cout << "V dims: " << V.rows() << "x" << V.cols() << "\n";
  std::cout << "T dims: " << T.rows() << "x" << T.cols() << "\n";
#endif
  Eigen::MatrixX<Float> id_T = (es.eigenvectors() * (t * es.eigenvalues().array().abs().sqrt()).
       unaryExpr([](Float x) {
           return x;
       }).matrix().asDiagonal() *
       es.eigenvectors().transpose());

  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * id_T * e1;
}

template <typename Float, typename F>
void gautschi_step_krylov(
    Eigen::VectorX<Float> &unext, Eigen::VectorX<Float> &vnext,
    const Eigen::VectorX<Float> &un, const Eigen::VectorX<Float> &vn,
    const Eigen::VectorX<Float> &upast, Eigen::VectorX<Float> &ubuf,
    const Eigen::SparseMatrix<Float> &Laplacian, const uint32_t nx,
    const uint32_t ny, const Float dx, const Float dy, const Float tau, F fun) {

  ubuf = Omega_scaled(Laplacian, un, tau);
#if DEBUG
  std::cout << ubuf << "\n";;
#endif

  ubuf = ubuf.array().unaryExpr(fun);
  unext = 2 * cos_L(Laplacian, un, tau) - upast +
          tau * tau * sinc2_L_half(Laplacian, ubuf, tau/2);
  vnext = (unext - upast) / 2 / tau;
}

template <typename Float, typename F>
void gautschi_step_almohy(Eigen::VectorX<Float> &unext,
                          Eigen::VectorX<Float> &vnext,
                          const Eigen::VectorX<Float> &un,
                          const Eigen::VectorX<Float> &vn,
                          const Eigen::VectorX<Float> &upast,
                          Eigen::VectorX<Float> &ubuf,
                          const Eigen::SparseMatrix<Float> &Laplacian,
                          const uint32_t nx, const uint32_t ny, const Float dx,
                          const Float dy, const Float tau, F fun) {}

// We have some assumptions in this function:
// write u_tt = u_xx + u_yy + f(u) (f(u) != f(u, x, y, t))
// - no nonlinearity c (wave speed) or phi (focusing nonlinearity)
// - use heterogeneous von Neumann (no-flux) BCs

template <typename Float, typename StepFn, typename NF>
requires StepFunction<StepFn, Float, NF>
    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    evolve(StepFn step_fn, NF NonlinearFunc,
           const Eigen::VectorX<Float> &initial_u,
           const Eigen::VectorX<Float> &initial_v, const uint32_t nx,
           const uint32_t ny, const Float dx, const Float dy, const Float Lxmin,
           const Float Lxmax, const Float Lymin, const Float Lymax,
           const Float t0, const Float T, const uint32_t nt,
           const uint32_t num_snapshots) {
  const Float dt = (T - t0) / static_cast<Float>(nt);
  const uint32_t snapshot_frequency = nt / num_snapshots;

  Eigen::VectorX<Float> u_save(num_snapshots * nx * ny);
  Eigen::VectorX<Float> v_save(num_snapshots * nx * ny);

  const auto Laplacian = buildD2<Float>(nx - 2, ny - 2, dx, dy); 
  const Eigen::SparseMatrix<Float> NLaplacian = -Laplacian;

  Eigen::VectorX<Float> buf(nx * ny);

  Eigen::VectorX<Float> un = initial_u;
  Eigen::VectorX<Float> vn = initial_v;

  Eigen::VectorX<Float> unext(nx * ny);
  Eigen::VectorX<Float> vnext(nx * ny);

  Eigen::VectorX<Float> upast = un - dt * vn;

  Eigen::Map<Eigen::Matrix<Float, -1, -1, Eigen::RowMajor>>(
      u_save.data(), num_snapshots, nx * ny)
      .row(0) = un.transpose();

  Eigen::Map<Eigen::Matrix<Float, -1, -1, Eigen::RowMajor>>(
      v_save.data(), num_snapshots, nx * ny)
      .row(0) = vn.transpose();

  uint32_t curr_snapshot_index = 1;
  for (uint32_t ti = 1; ti < nt; ++ti) {
    step_fn(unext, vnext, un, vn, upast, buf, NLaplacian, nx, ny, dx, dy, dt,
            NonlinearFunc);
    upast = un;
    un = unext;
    vn = vnext;
    apply_neumann_bc(un, vn, nx, ny);
    if (ti % snapshot_frequency == 0) {
      Eigen::Map<Eigen::Matrix<Float, -1, -1, Eigen::RowMajor>>(
          u_save.data(), num_snapshots, nx * ny)
          .row(curr_snapshot_index) = un.transpose();
      Eigen::Map<Eigen::Matrix<Float, -1, -1, Eigen::RowMajor>>(
          v_save.data(), num_snapshots, nx * ny)
          .row(curr_snapshot_index) = vn.transpose();

      curr_snapshot_index++;
    }
#if !DEBUG
    PROGRESS_BAR(ti, nt);
#endif
  }
  return {u_save, v_save};
}

template <typename Float>
std::pair<Eigen::VectorX<Float>, Eigen::MatrixX<Float>>
analyze_spectrum(const Eigen::SparseMatrix<Float>& L) {
    Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<Float>> es(L);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("Eigenvalue computation failed");  
    return {es.eigenvalues(), es.eigenvectors()};
}

template <typename Float>
void soliton_evolution(const Float L, const uint32_t nx, const uint32_t ny) {
  const uint32_t nt = 10000;
  const uint32_t snapshots = 100;
  const Float t0 = 0.;
  const Float T = 1000.;

  const auto dx = 2 * L / nx;
  const auto dy = 2 * L / ny;

  auto soliton = [](Float x, Float y) {
    return 2. * std::atan(std::exp(3. - 5.1 * std::sqrt(x * x + y * y)));
  };

  std::mt19937 gen(std::random_device{}());
  Float mean = .1;
  Float stddev = 2.;

  std::normal_distribution<float> dist(mean, stddev);
  auto rand_field = [&](Float x, Float y) {
    return dist(gen);
  };

  auto zero_everywhere = [](Float x, Float y) { return 0.; };

  auto semilinear_term = [](Float ui) { return -std::sin(ui); };

  const auto initial_u = apply_function_uniform(-L, L, nx, -L, L, ny, rand_field);
  const auto initial_v =
      apply_function_uniform(-L, L, nx, -L, L, ny, zero_everywhere);

  using NonlinearFuncType = decltype(semilinear_term);
  //auto step = stormer_verlet_step_spmv<Float, NonlinearFuncType>;
  auto step = gautschi_step_krylov<Float, NonlinearFuncType>;

  const auto [u_history, v_history] =
      evolve(step, semilinear_term, initial_u, initial_v, nx, ny, dx, dy, -L, L,
             -L, L, t0, T, nt, snapshots);

  const std::vector<uint32_t> shape = {snapshots, nx, ny};
  const auto fname_u = "/home/konrad/code/msc-thesis/from-scratch/evolution_u_gautschi_krylov.npy"; 
  //const auto fname_u = "/home/konrad/code/msc-thesis/from-scratch/evolution_u_sv_spmv.npy";

  
  const auto fname_v = "/home/konrad/code/msc-thesis/from-scratch/evolution_v_gautschi_krylov.npy";
  //const auto fname_v = "/home/konrad/code/msc-thesis/from-scratch/evolution_v_sv_spmv.npy";

  save_to_npy(fname_u, u_history, shape);
  save_to_npy(fname_v, u_history, shape);
}

void analyze_lapl() {
  uint32_t nx, ny;
  nx = ny = 10;
  double L = 3.;
  const auto dx = 2 * L / nx;
  const auto dy = 2 * L / ny;
  const auto Laplacian = buildD2<double>(nx - 2, ny - 2, dx, dy);
  const Eigen::SparseMatrix<double> NLaplacian = -Laplacian;
  
  const auto [ev, vectors] = analyze_spectrum(Laplacian);
  const auto [nev, nvectors] = analyze_spectrum(NLaplacian);

  std::cout << "L: " << ev << "\n";
  std::cout << "-L: " << nev << "\n";
}



int main() {
  //analyze_lapl();
  soliton_evolution<double>(4., 32, 32);
  // benchmark();
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
