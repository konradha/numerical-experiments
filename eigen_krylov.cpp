// clang++ -std=c++20 eigen_krylov.cpp -I$CONDA_PREFIX/include/eigen3 -Wl,-rpath,$CONDA_PREFIX/lib -o eigen-solver

#include "npy.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <string>
#include <vector>

template <typename Float>
Eigen::SparseMatrix<double> buildD2(uint32_t nx, uint32_t ny, Float dx, Float dy) {
    assert(nx == ny);
    assert(std::abs(dx - dy) < 1e-10);

    const uint32_t N = (nx + 2) * (nx + 2);
    const uint32_t nnz = N + 4*(N - 1) - 4;
    using T = Eigen::Triplet<Float>;
    std::vector<T> triplets;
    triplets.reserve(nnz);

    for (uint32_t i = 0; i < N; ++i) {
        Float val = static_cast<Float>(-4.0);
        if (i < (nx + 2) || i >= N - (nx + 2) ||
            i % (nx + 2) == 0 || i % (nx + 2) == (nx + 1)) {
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
    L *= 1.0 / (dx * dy);
    return L;
}

template<typename Float, typename F>
Eigen::VectorX<Float> apply_function(
    const Eigen::VectorX<Float>& x,
    const Eigen::VectorX<Float>& y,
    F f
) {
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


template<typename Float, typename F>
Eigen::VectorX<Float> apply_function_uniform(
    Float x_min, Float x_max, uint32_t nx,
    Float y_min, Float y_max, uint32_t ny,
    F f
) {
    Eigen::VectorX<Float> x = Eigen::VectorX<Float>::LinSpaced(nx, x_min, x_max);
    Eigen::VectorX<Float> y = Eigen::VectorX<Float>::LinSpaced(ny, y_min, y_max);
    return apply_function<Float>(x, y, f);
}

template<typename Float>
void save_to_npy(const std::string& filename, 
                 const Eigen::VectorX<Float>& data,
                 const std::vector<uint32_t>& shape) {
    std::vector<Float> vec(data.data(), data.data() + data.size());
    std::vector<uint64_t> shape_ul;
    for(const auto dim : shape)
        shape_ul.push_back(static_cast<uint64_t>(dim));
    for(const auto dim : shape)
        std::cout << dim << " ";
    std::cout << "\n" << "shape size: " << shape.size() << "\n";
    
    npy::SaveArrayAsNumpy(filename, false, shape.size(), shape_ul.data(), vec);
}

int main() {
    uint32_t nx = 100;
    uint32_t ny = 100;

    double L = 3;
    double dx = 2 * L / nx;
    double dy = 2 * L / ny;

    const auto Laplacian = buildD2<double>(nx, ny, dx, dy);

    auto gaussian = [] (double x, double y) {
        return std::exp(-(x * x + y * y));
    };
    const std::vector<uint32_t> shape = {nx, ny};
    auto u = apply_function_uniform(-L, L, nx, -L, L, ny, gaussian);

    auto del_u = Laplacian * u;


    const auto fname = "/home/konrad/code/msc-thesis/from-scratch/outfile.npy";
    save_to_npy(fname, u, shape);
}
