//==--- fluidity/tests/reconstruction_tests.cpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reconstruction_tests.cpp
/// \brief This file defines tests for data reconstruction.
//
//==------------------------------------------------------------------------==//

#include <fluidity/algorithm/fill.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/iterator/range.hpp>
#include <fluidity/limiting/limiters.hpp>
#include <fluidity/material/ideal_gas.hpp>
#include <fluidity/reconstruction/muscl_reconstructor.hpp>
#include <fluidity/state/state.hpp>
#include <gtest/gtest.h>

using namespace fluid;

// Defines the type of data to use.
using real_t         = double;
// Defines a 1 dimensional primitive state.
using state_1d_t     = state::primitive_t<real_t, 1>;
// Defines the type of the container for the states.
using container_1d_t = HostTensor<state_1d_t, 1>;

// Fixture class for all 1D reconstructor tests.
class muscl_reconstruction_1d_tests : public ::testing::Test {
 protected:
  // Defines the type of data to use for the tests.
  using real_t      = double;
  // Defines a 1 dimensional primitive state.
  using state_t     = state::primitive_t<real_t, 1>;
  // Defines the type of the container for the states.
  using container_t = HostTensor<state_1d_t, 1>;

  // Defines the type of the reconstructor object with a limiter.
  template <typename Limiter>
  using reconstructor_t = recon::MHReconstructor<real_t, Limiter>;

  //==--- Materials --------------------------------------------------------==//
  using ideal_gas_t = material::IdealGas<real_t>;
  //==----------------------------------------------------------------------==//

  /// Defines the number of states to use.
  static constexpr auto num_states = std::size_t{5};

  // The setup involves creating default data for the tests, each test can
  // then modify the appropriate states for the test.
  virtual void SetUp()
  {
    states.resize(num_states);
    for (auto& state : states)
    {
      state.set_density(1.0);
      state.set_pressure(1.0);
      state.set_velocity(0.125, dim_x);
    }
  }
  container_t states; // state data for the reconstruction.
};

// Tests the the muscl reconstructor returns the same left and right states
// when the left and right input states are the same.
TEST_F(muscl_reconstruction_1d_tests, constant_state_data)
{
  // Create a reconstructor with no limting:
  auto reconstructor = reconstructor_t<limit::Linear>();
  auto material      = ideal_gas_t();

  for (const auto& state : states)
  {
    EXPECT_EQ(state.density()         , 1.000);
    EXPECT_EQ(state.pressure(material), 1.000);
    EXPECT_EQ(state.velocity(dim_x)   , 0.125);
  }

  for (auto offset : range(std::size_t{1}, states.size() - 2))
  {
    auto state_it = states.multi_iterator().offset(offset, dim_x);
    auto recon    = reconstructor(state_it, material, 1.0, dim_x);

    for (auto i : range(state_it->size()))
    {
      EXPECT_EQ(recon.left[i], recon.right[i]);
    }
  }
}

TEST_F(muscl_reconstruction_1d_tests, linear_limiting)
{
  // Create a reconstructor with no limting:
  auto reconstructor = reconstructor_t<limit::Linear>();
  auto material      = ideal_gas_t();

  // Get the left input state:
  constexpr auto left_offset = 2;
  constexpr auto dtdh        = real_t{1.0};
  auto left  = states.multi_iterator().offset(left_offset, dim_x);
  auto right = left.offset(1, dim_x);

  // Modify the right input state data:
  for (const auto offset : range(std::size_t{0}, states.size() - left_offset - 1))
  {
    auto state = right.offset(offset, dim_x);
    state->set_density(0.125);
    state->set_pressure(0.1);
    state->set_velocity(0.0, dim_x);

    EXPECT_EQ(state->density()         , 0.125);
    EXPECT_EQ(state->pressure(material), 0.100);
    EXPECT_EQ(state->velocity(dim_x)   , 0.000);
  }
  // Perform the reconstruction:
  auto  recon = reconstructor(left, material, dtdh, dim_x);

  // Test the results:
  constexpr auto precision = 0.00001;

  auto& state = recon.left;
  auto d_density  = std::abs(real_t{0.83984} - state.density());
  auto d_pressure = std::abs(real_t{0.98381} - state.pressure(material));
  auto d_velocity = std::abs(real_t{0.33019} - state.velocity(dim_x));

  EXPECT_LT(d_density , precision);
  EXPECT_LT(d_pressure, precision);
  EXPECT_LT(d_velocity, precision);

  state = recon.right;
  d_density  = std::abs(real_t{0.34766} - state.density());
  d_pressure = std::abs(real_t{0.33594} - state.pressure(material));
  d_velocity = std::abs(real_t{0.25646} - state.velocity(dim_x));

  EXPECT_LT(d_density , precision);
  EXPECT_LT(d_pressure, precision);
  EXPECT_LT(d_velocity, precision);
}

int main(int argc, char** argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}