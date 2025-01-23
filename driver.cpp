// nvcc -std=c++17 --expt-relaxed-constexpr -lcusparse -x cu --extended-lambda -arch=sm_70 driver.cpp -o sg_driver

#include "sg_stepper.h"
#include <cmath>
#include <iostream>

int main() {
   const uint32_t nx = 256, ny = 256;
   const double dx = 0.1, dy = 0.1;
   const double dt = 0.01;
   const double Lx = nx * dx, Ly = ny * dy;
   
   auto L = constructLaplacian2D<double>(nx, ny, dx, dy);
   
   Eigen::VectorXd c = Eigen::VectorXd::Ones(nx * ny);
   Eigen::VectorXd m = Eigen::VectorXd::Ones(nx * ny);
   
   Eigen::VectorXd u0(nx * ny);
   Eigen::VectorXd v0 = Eigen::VectorXd::Zero(nx * ny);
   
   for(uint32_t i = 0; i < ny; ++i) {
       for(uint32_t j = 0; j < nx; ++j) {
           double x = dx * j - Lx/2;
           double y = dy * i - Ly/2;
           double r = std::sqrt(x*x + y*y);
           u0[i*nx + j] = 4.0 * std::atan(exp(2.0 - r));
       }
   }
   
   SineGordonCPU cpu_solver(nx, ny, dx, dy, L, m, c);
   SineGordonGPU gpu_solver(nx, ny, dx, dy, L, m, c);
   
   cpu_solver.setState(u0, v0, dt);
   gpu_solver.setState(u0, v0, dt);
   
   cpu_solver.step(dt);
   gpu_solver.step(dt);
   
   Eigen::VectorXd u_cpu, v_cpu, u_gpu, v_gpu;
   cpu_solver.getState(u_cpu, v_cpu);
   gpu_solver.getState(u_gpu, v_gpu);
   
   double u_diff = (u_cpu - u_gpu).norm() / u_cpu.norm();
   double v_diff = (v_cpu - v_gpu).norm() / v_cpu.norm();
   
   std::cout << "Relative difference in u: " << u_diff << std::endl;
   std::cout << "Relative difference in v: " << v_diff << std::endl;
   
   return 0;
}

