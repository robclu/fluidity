//==--- fluidity/tests/state_tests_host.cpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_tests_host.cpp
/// \brief This file defines tests for host side state operations.
//
//==------------------------------------------------------------------------==//

#include <gtest/gtest.h>
#include <fluidity/material/ideal_gas.hpp>
#include <fluidity/state/state.hpp>

/// Defines the type of data to use.
using real_t           = double;
/// Defines a 2 dimensional conservative state with one additional component.
using conservative2d_t = fluid::state::conservative_t<real_t, 2, 1>;
/// Defines a 2 dimensional conservative state with one additional component.
using primitive2d_t    = fluid::state::primitive_t<real_t, 2, 1>;
/// Defines the material type to use for the tests.
using material_t       = fluid::material::IdealGas<real_t>;

/// Defines the adiabatic index of an ideal gas.
static constexpr real_t adi_index = 1.4;

using namespace fluid::state;

TEST(state_tests_host, canCreateprim_state) {
  primitive2d_t state;
  material_t    material(adi_index);

  state.set_density(real_t(1));
  state.set_pressure(real_t(2));
  state.set_velocity(real_t(3), fluid::dim_x);
  state.set_velocity(real_t(4), fluid::dim_y);
  state.set_additional(real_t(5), 0);

  EXPECT_EQ(state.density()             , real_t(1));
  EXPECT_EQ(state.pressure(material)    , real_t(2));
  EXPECT_EQ(state.velocity(fluid::dim_x), real_t(3));
  EXPECT_EQ(state.velocity(fluid::dim_y), real_t(4));
  EXPECT_EQ(state.additional(0)         , real_t(5));
}

TEST(state_tests_host, canCreatecons_state) {
  conservative2d_t state;
  material_t       material(adi_index);

  state.set_density(real_t(1));
  state.set_velocity(real_t(2), fluid::dim_x);
  state.set_velocity(real_t(3), fluid::dim_y);
  state.set_energy(real_t(4));
  state.set_additional(real_t(5), 0);

  EXPECT_EQ(state.density()             , real_t(1));
  EXPECT_EQ(state.velocity(fluid::dim_x), real_t(2));
  EXPECT_EQ(state.velocity(fluid::dim_y), real_t(3));
  EXPECT_EQ(state.energy(material)      , real_t(4));
  EXPECT_EQ(state.additional(0)         , real_t(5));
}


TEST(state_tests_host, canConvertPrimitiveToConservative) {
  primitive2d_t prim_state;
  material_t    material(adi_index);

  prim_state.set_density(1.0);
  prim_state.set_pressure(2.0);
  prim_state.set_velocity(3.0, fluid::dim_x);
  prim_state.set_velocity(4.0, fluid::dim_y);
  prim_state.set_additional(5.0, 0);

  EXPECT_EQ(prim_state.density()             , real_t(1.0));
  EXPECT_EQ(prim_state.pressure(material)    , real_t(2.0));
  EXPECT_EQ(prim_state.velocity(fluid::dim_x), real_t(3.0));
  EXPECT_EQ(prim_state.velocity(fluid::dim_y), real_t(4.0));
  EXPECT_EQ(prim_state.additional(0)         , real_t(5.0));

  auto cons_state = prim_state.conservative(material);
  auto conv_state = cons_state.primitive(material);

  EXPECT_EQ(conv_state.density()             , real_t(1.0));
  EXPECT_EQ(conv_state.velocity(fluid::dim_x), real_t(3.0));
  EXPECT_EQ(conv_state.velocity(fluid::dim_y), real_t(4.0));
  EXPECT_EQ(conv_state.additional(0)         , real_t(5.0));

  EXPECT_EQ(prim_state.density()             , cons_state.density()             );
  EXPECT_EQ(prim_state.velocity(fluid::dim_x), cons_state.velocity(fluid::dim_x));
  EXPECT_EQ(prim_state.velocity(fluid::dim_y), cons_state.velocity(fluid::dim_y));
  EXPECT_EQ(prim_state.energy(material)      , cons_state.energy(material)      );

  constexpr auto tolerance = 1e-15;
  const auto diff_a = std::abs(conv_state.pressure(material) - real_t(2.0));
  const auto diff_b = std::abs(prim_state.pressure(material) -
                               cons_state.pressure(material));

  EXPECT_LT(diff_a, tolerance);
  EXPECT_LT(diff_b, tolerance);
}

TEST(state_tests_host, canConvertConservativeToPrimitive) {
  conservative2d_t cons_state;
  material_t       material(adi_index);

  cons_state.set_density(1.0);
  cons_state.set_velocity(2.0, fluid::dim_x);
  cons_state.set_velocity(2.5, fluid::dim_y);
  cons_state.set_energy(3.0);
  cons_state.set_additional(3.0, 0);

  EXPECT_EQ(cons_state.density()             , real_t(1.0));
  EXPECT_EQ(cons_state.velocity(fluid::dim_x), real_t(2.0));
  EXPECT_EQ(cons_state.velocity(fluid::dim_y), real_t(2.5));
  EXPECT_EQ(cons_state.energy(material)      , real_t(3.0));
  EXPECT_EQ(cons_state.additional(0)         , real_t(3.0));

  auto prim_state = cons_state.primitive(material);
  auto conv_state = prim_state.conservative(material);

  EXPECT_EQ(conv_state.density()             , real_t(1.0));
  EXPECT_EQ(conv_state.velocity(fluid::dim_x), real_t(2.0));
  EXPECT_EQ(conv_state.velocity(fluid::dim_y), real_t(2.5));
  EXPECT_EQ(conv_state.energy(material)      , real_t(3.0));
  EXPECT_EQ(conv_state.additional(0)         , real_t(3.0));

  EXPECT_EQ(prim_state.density()             , cons_state.density()             );
  EXPECT_EQ(prim_state.velocity(fluid::dim_x), cons_state.velocity(fluid::dim_x));
  EXPECT_EQ(prim_state.velocity(fluid::dim_y), cons_state.velocity(fluid::dim_y));
  EXPECT_EQ(prim_state.pressure(material)    , cons_state.pressure(material)    );
  EXPECT_EQ(prim_state.energy(material)      , cons_state.energy(material)      );
}

TEST(state_tests_host, primitiveAndConservativeFluxesAreTheSame) {
  conservative2d_t cons_state;
  material_t       material(adi_index);

  const real_t density  = 1.0,
               v_x      = 2.0,
               v_y      = 2.5,
               energy   = 3.0,
               add_comp = 3.0,
               pressure = (adi_index - real_t(1)) 
                        * (energy
                        -  real_t(0.5) * density * (v_x * v_x + v_y * v_y));

  cons_state.set_density(density);
  cons_state.set_velocity(v_x, fluid::dim_x);
  cons_state.set_velocity(v_y, fluid::dim_y);
  cons_state.set_energy(energy);
  cons_state.set_additional(add_comp, 0);

  EXPECT_EQ(cons_state.density()             , density);
  EXPECT_EQ(cons_state.velocity(fluid::dim_x), v_x);
  EXPECT_EQ(cons_state.velocity(fluid::dim_y), v_y);
  EXPECT_EQ(cons_state.energy(material)      , energy);
  EXPECT_EQ(cons_state.additional(0)         , add_comp);

  auto prim_state  = cons_state.primitive(material);
  auto cons_fluxes = cons_state.flux(material, fluid::dim_x);
  auto prim_fluxes = prim_state.flux(material, fluid::dim_x);
  
  using index_t = typename conservative2d_t::index;
  const auto di      = index_t::density;
  const auto vxi     = index_t::velocity(fluid::dim_x);
  const auto vyi     = index_t::velocity(fluid::dim_y);
  const auto ei      = index_t::energy;
  const auto ai      = index_t::additional(0);

  // Hand calculations:
  EXPECT_EQ(cons_fluxes[di] , density * v_x                 ); // rho * u
  EXPECT_EQ(cons_fluxes[ei] , v_x * (energy + pressure)     ); // u * (E + p)
  EXPECT_EQ(cons_fluxes[vxi], density * v_x * v_x + pressure); // rho * u * u + p
  EXPECT_EQ(cons_fluxes[vyi], density * v_x * v_y           ); // rho * u * v
  EXPECT_EQ(cons_fluxes[ai] , density * v_x * add_comp      ); // rho * u * additional

  EXPECT_EQ(cons_fluxes[di] , prim_fluxes[di] );
  EXPECT_EQ(cons_fluxes[ei] , prim_fluxes[ei] );
  EXPECT_EQ(cons_fluxes[vxi], prim_fluxes[vxi]);
  EXPECT_EQ(cons_fluxes[vyi], prim_fluxes[vyi]);
  EXPECT_EQ(cons_fluxes[ai] , prim_fluxes[ai] );
}

/*
TEST(StateTests, CanCreateStatesFromVectorExpressions) {
  Flow::Vec<T, 5> u(T(1), T(2), T(3), T(4), T(5));
  Flow::Vec<T, 5> v(T(1), T(2), T(3), T(4), T(5)); 

  Conservative2D c = u + v;

  EXPECT_EQ(c[0], T(2) );
  EXPECT_EQ(c[1], T(4) );
  EXPECT_EQ(c[2], T(6) );
  EXPECT_EQ(c[3], T(8) );
  EXPECT_EQ(c[4], T(10));
}
*/

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}