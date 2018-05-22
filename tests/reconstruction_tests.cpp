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
#include <fluidity/limiting/van_leer_limiter.hpp>
#include <fluidity/reconstruction/muscl_reconstructor.hpp>
#include <fluidity/state/state.hpp>
#include <gtest/gtest.h>

using namespace fluid;

// Defines the type of data to use.
using real_t      = double;
// Defines a 1 dimensional primitive state.
using state_t     = state::primitive_t<real_t, 1>;
// Defines the type of the container for the states.
using container_t = HostTensor<state_t, 1>;

// This test uses a limiter which is known to work correctly.
TEST(reconstruction, muscle_reconstructor_same_states)
{
  // Defines the type of the reconstructor for the test:
  using reconstructor_t = recon::MHReconstructor<real_t, limit::VanLeer>;
  container_t states(5);

  fill(states.begin(), states.end(), [] (auto& state)
  {
    state = state_t{ 0.1, 1.0, 0.125 };
  });

  for (const auto& state : states)
  {
    EXPECT_EQ(state[0], 0.100);
    EXPECT_EQ(state[1], 1.000);
    EXPECT_EQ(state[2], 0.125);
  }
}

int main(int argc, char** argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}