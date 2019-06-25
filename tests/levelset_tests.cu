//==--- fluidity/tests/levelset_tests.cu ------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_tests.cu
/// \brief This file defines tests for levelset functionality for fluiidity.
//
//==------------------------------------------------------------------------==//

#include <gtest/gtest.h>
#include <fluidity/levelset/levelset.hpp>

// Define the data type for the levelset.
using real_t = double;

// Define the type of a 1D levelset.
using levelset_1d_t = fluid::LevelSet<real_t, 1, fluid::exec::DeviceKind::gpu>;

TEST(levelset, can_initialize_levelset_correctly)
{
  levelset_1d_t levelset(
    [] fluidity_host_device (auto it, auto& positions)
    {
      *it = positions[0] < 0.25 ? real_t{0} : real_t{1}; 
    }, 20
  );
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}