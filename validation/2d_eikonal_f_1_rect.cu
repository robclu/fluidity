//==--- fluidity/validation/2d_eikonal_f_1_rect.cu --------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  2d_eikonal_f_1_rect.cu
/// \brief This file defines a two dimensional validation case for the Eikonal
///        solver where the speed function has speed f=1, and the source node
///        is places in the centre of the domain and is a rectangle.
//
//==------------------------------------------------------------------------==//

#include <fluidity/algorithm/fill.hpp>
#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/vec.hpp>
#include <fluidity/geometry/rectangle.hpp>
#include <fluidity/scheme/eikonal/fast_iterative.hpp>
#include <fluidity/solver/eikonal_solver.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>

using namespace fluid;

using real_t = double;
using vec2_t = Vec2<real_t>;
using pos_t  = Vec2<real_t>;

static constexpr auto size_x = int{2000};
static constexpr auto size_y = int{2000};
static constexpr auto dims   = 2;
static constexpr auto res    = 1.0;

template <typename I, typename T, typename O>
auto output_data(I&& it, T d, O& output) -> void {
  if (d == 0) {
    for (auto i : range(it.size(d))) {
      output
        << std::setw(6)
        << std::setprecision(4)
        << std::scientific
        << *it.offset(i, d)
        << " ";
    }
    output << "\n";
  } else {
    for (auto i : range(it.size(d))) {
      output_data(it, d - 1, output);
      it.shift(1, d);
    }
    std::cout << "\n";
  }
}

template <typename I, typename T>
void write_data(I& it, T d) {
  std::ofstream output_file;
  auto filename = "box_2d_output.dat";
  output_file.open(filename, std::fstream::trunc);
  output_data(std::forward<I>(it), d, output_file);
  output_file.close();
}

template <typename I, typename T>
auto print_data(I&& it, T d) -> void {
  if (d == 0) {
    for (auto i : range(it.size(d))) {
      std::cout
        << std::setprecision(2)
        << std::scientific
        << *it.offset(i, d)
        << " ";
    }
    std::cout << "\n";
  } else {
    for (auto i : range(it.size(d))) {
      print_data(it, d - 1);
      it.shift(1, d);
    }
    std::cout << "\n";
  }
}

int main(int argc, char** argv) {
  // What we would ideally do (if this was optimized), is to create a context
  // with a device type for the system (this should be able to be determined
  // quite easily), which is either GPU, CPU, or defauly (which would choose
  // the optimal one).
  // auto context = Context::default();
  
  // TODO: Finish this example with the ideal interface ...

  // The test is going to be run on the device, so we use a device tensor.
  using storage_t = DeviceTensor<real_t, 2>;
  auto input = storage_t{size_x, size_y};

  // Fill the input data, we set each cell as the signed distance from the
  // center of the domain. Since everything is outside of the center cell, the
  // signed distance for all cells is positive.
  fill(input.multi_iterator(), [&] fluidity_host_device (auto& cell)
  {
    constexpr auto center_x = static_cast<real_t>(size_x) / 2.0;
    constexpr auto center_y = static_cast<real_t>(size_y) / 2.0;
    constexpr auto length_x = static_cast<real_t>(size_x) / 2.0;
    constexpr auto length_y = static_cast<real_t>(size_y) / 2.0;

    using namespace geometry;
    auto p = pos_t(flattened_id(dim_x), flattened_id(dim_y));

    *cell = Rect<real_t>{
      vec2_t(length_x, length_y), pos_t{center_x, center_y}
    }.distance(p);
  });

  // Create the output data from the input data. We don't care about the data
  // in the output tensor, so we just copy the metadata.
  auto output = input.copy_without_data();

  // Since this test uses a constant speed function, we do not need to create
  // speed data for the solver, and can just solve for the input data.
  solver::eikonal(input, output, res, scheme::eikonal::FastIterative());

  auto host_out    = output.as_host();
  auto host_out_it = host_out.multi_iterator();
  using iter_t     = std::decay_t<decltype(host_out_it)>;

  // Print the results ...
  auto outer_dim = iter_t::dimensions - 1;
  write_data(host_out_it, iter_t::dimensions - 1);
}
