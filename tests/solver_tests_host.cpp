//==--- fluidity/tests/solver_tests_host.cpp --------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  solver_tests_host.cpp
/// \brief This file defines tests for host side solver related functionality.
//
//==------------------------------------------------------------------------==//

#include <fluidity/algorithm/fill.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/solver/boundary_loader.hpp>
#include <fluidity/state/state.hpp>
#include <gtest/gtest.h>

using namespace fluid;
using namespace fluid::solver;

// Defines a 1-dimensional conservative state with one additional component.
using primitive1d_t = state::primitive_t<float, 1>;

TEST(boundry_loader, can_load_transmissive_boundaries_1d)
{
  host_tensor1d<primitive1d_t> data(8);
  float c   = 0.0f;
  float inc = 1.0f;
  fill(data.begin(), data.end(), [&c, inc] (auto& e)
  {
    e.set_density(1);
    e.set_velocity(c, dim_x);
    c += inc;
  });

  BoundarySetter setter;
  setter.configure(dim_x, BoundaryIndex::first , BoundaryKind::transmissive);
  setter.configure(dim_x, BoundaryIndex::second, BoundaryKind::transmissive);

  setter(*(data.begin() + 3), *(data.begin())    , dim_x, BoundaryIndex::first);
  setter(*(data.begin() + 2), *(data.begin() + 1), dim_x, BoundaryIndex::first);

  setter(*(data.end() - 4), *(data.end() - 1), dim_x, BoundaryIndex::second);
  setter(*(data.end() - 3), *(data.end() - 2), dim_x, BoundaryIndex::second);

  for (const auto& e : data)
  {
    EXPECT_EQ(e.density(), 1);
  }

  EXPECT_EQ(data.begin()->velocity(dim_x)      , 3.0f);
  EXPECT_EQ((data.begin() + 1)->velocity(dim_x), 2.0f);
  EXPECT_EQ((data.begin() + 2)->velocity(dim_x), 2.0f);
  EXPECT_EQ((data.begin() + 3)->velocity(dim_x), 3.0f);

  EXPECT_EQ((data.end() - 1)->velocity(dim_x), 4.0f);
  EXPECT_EQ((data.end() - 2)->velocity(dim_x), 5.0f);
  EXPECT_EQ((data.end() - 3)->velocity(dim_x), 5.0f);
  EXPECT_EQ((data.end() - 4)->velocity(dim_x), 4.0f);
}

TEST(boundry_loader, can_load_reflective_boundaries_1d)
{
  host_tensor1d<primitive1d_t> data(8);
  float c   = 0.0f;
  float inc = 1.0f;
  fill(data.begin(), data.end(), [&c, inc] (auto& e)
  {
    e.set_density(1);
    e.set_velocity(c, dim_x);
    c += inc;
  });

  BoundarySetter setter;
  setter.configure(dim_x, BoundaryIndex::first , BoundaryKind::reflective);
  setter.configure(dim_x, BoundaryIndex::second, BoundaryKind::reflective);

  setter(*(data.begin() + 3), *(data.begin())    , dim_x, BoundaryIndex::first);
  setter(*(data.begin() + 2), *(data.begin() + 1), dim_x, BoundaryIndex::first);

  setter(*(data.end() - 4), *(data.end() - 1), dim_x, BoundaryIndex::second);
  setter(*(data.end() - 3), *(data.end() - 2), dim_x, BoundaryIndex::second);

  for (const auto& e : data)
  {
    EXPECT_EQ(e.density(), 1);
  }

  EXPECT_EQ(data.begin()->velocity(dim_x)      , -3.0f);
  EXPECT_EQ((data.begin() + 1)->velocity(dim_x), -2.0f);
  EXPECT_EQ((data.begin() + 2)->velocity(dim_x),  2.0f);
  EXPECT_EQ((data.begin() + 3)->velocity(dim_x),  3.0f);

  EXPECT_EQ((data.end() - 1)->velocity(dim_x), -4.0f);
  EXPECT_EQ((data.end() - 2)->velocity(dim_x), -5.0f);
  EXPECT_EQ((data.end() - 3)->velocity(dim_x),  5.0f);
  EXPECT_EQ((data.end() - 4)->velocity(dim_x),  4.0f);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}