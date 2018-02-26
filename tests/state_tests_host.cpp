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
using conservative2d_t = fluid::state::conservative_t<real_t, 2>;
/// Defines a 2 dimensional conservative state with one additional component.
using primitive2d_t    = fluid::state::primitive_t<real_t, 2>;
/// Defines the material type to use for the tests.
using material_t       = fluid::material::IdealGas<real_t>;

/// Defines the adiabatic index of an ideal gas.
static constexpr real_t adi_index = 1.4;

using namespace fluid::state;

TEST(state_tests_host, canCreatePrimitiveState) {
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
/*
TEST(StateTests, CanCreateConservativeState) {
  Conservative2D state;
  Gas            gas(adiIndex);

  state.density()      = T(1.0);
  state.rhoVelocity(0) = T(2.0);
  state.rhoVelocity(1) = T(2.5);
  state.energy()       = T(3.0);
  state.additional(0)  = T(3.0);

  EXPECT_EQ(state.density()     , T(1.0));
  EXPECT_EQ(state.rhoVelocity(0), T(2.0));
  EXPECT_EQ(state.rhoVelocity(1), T(2.5));
  EXPECT_EQ(state.energy()      , T(3.0));
  EXPECT_EQ(state.additional(0) , T(3.0));
}

TEST(StateTests, CanConvertPrimitiveToConservative) {
  Primitive2D primitiveState;
  Gas         gas(adiIndex);

  primitiveState.density()     = T(1.0);
  primitiveState.pressure()    = T(2.0);
  primitiveState.velocity(0)   = T(3.0);
  primitiveState.velocity(1)   = T(4.0);
  primitiveState.additional(0) = T(5.0);

  EXPECT_EQ(primitiveState.density()    , T(1.0));
  EXPECT_EQ(primitiveState.pressure()   , T(2.0));
  EXPECT_EQ(primitiveState.velocity(0)  , T(3.0));
  EXPECT_EQ(primitiveState.velocity(1)  , T(4.0));
  EXPECT_EQ(primitiveState.additional(0), T(5.0));

  auto conservativeState = primitiveState.conservative(gas);
  auto convertedState    = conservativeState.conservative(gas);

  EXPECT_EQ(convertedState.density()     , T(1.0));
  EXPECT_EQ(convertedState.velocity(0)   , T(3.0));
  EXPECT_EQ(convertedState.velocity(1)   , T(4.0));
  EXPECT_EQ(convertedState.additional(0) , T(5.0));

  EXPECT_EQ(primitiveState.density()    , conservativeState.density()    );
  EXPECT_EQ(primitiveState.velocity(0)  , conservativeState.velocity(0)  );
  EXPECT_EQ(primitiveState.velocity(1)  , conservativeState.velocity(1)  );
  EXPECT_EQ(primitiveState.energy(gas)  , conservativeState.energy()     );

  constexpr auto tolerance = 1e-15;
  const auto diffa         = std::abs(convertedState.pressure(gas) - T(2.0));
  const auto diffb         = std::abs(primitiveState.pressure() -
                              conservativeState.pressure(gas));

  EXPECT_LT(diffa, tolerance);
  EXPECT_LT(diffb, tolerance);
}


TEST(StateTests, CanConvertConservativeToPrimitive) {
  Conservative2D conservativeState;
  Gas            gas(adiIndex);

  conservativeState.density()      = T(1.0);
  conservativeState.rhoVelocity(0) = T(2.0);
  conservativeState.rhoVelocity(1) = T(2.5);
  conservativeState.energy()       = T(3.0);
  conservativeState.additional(0)  = T(3.0);

  EXPECT_EQ(conservativeState.density()     , T(1.0));
  EXPECT_EQ(conservativeState.rhoVelocity(0), T(2.0));
  EXPECT_EQ(conservativeState.rhoVelocity(1), T(2.5));
  EXPECT_EQ(conservativeState.energy()      , T(3.0));
  EXPECT_EQ(conservativeState.additional(0) , T(3.0));

  auto primitiveState = conservativeState.primitive(gas);
  auto convertedState = primitiveState.conservative(gas);

  EXPECT_EQ(convertedState.density()     , T(1.0));
  EXPECT_EQ(convertedState.rhoVelocity(0), T(2.0));
  EXPECT_EQ(convertedState.rhoVelocity(1), T(2.5));
  EXPECT_EQ(convertedState.energy(gas)   , T(3.0));
  EXPECT_EQ(convertedState.additional(0) , T(3.0));

  EXPECT_EQ(primitiveState.density()    , conservativeState.density()    );
  EXPECT_EQ(primitiveState.velocity(0)  , conservativeState.velocity(0)  );
  EXPECT_EQ(primitiveState.velocity(1)  , conservativeState.velocity(1)  );
  EXPECT_EQ(primitiveState.pressure(gas), conservativeState.pressure(gas));
  EXPECT_EQ(primitiveState.energy(gas)  , conservativeState.energy(gas)  );
}

TEST(StateTests, PrimitiveAndConservativeFluxesAreTheSame) {
  Conservative2D conservativeState;
  Gas            gas(adiIndex);

  conservativeState.density()      = T(1.0);
  conservativeState.rhoVelocity(0) = T(2.0);
  conservativeState.rhoVelocity(1) = T(2.5);
  conservativeState.energy()       = T(3.0);
  conservativeState.additional(0)  = T(3.0);

  EXPECT_EQ(conservativeState.density()     , T(1.0));
  EXPECT_EQ(conservativeState.rhoVelocity(0), T(2.0));
  EXPECT_EQ(conservativeState.rhoVelocity(1), T(2.5));
  EXPECT_EQ(conservativeState.energy()      , T(3.0));
  EXPECT_EQ(conservativeState.additional(0) , T(3.0));

  auto primitiveState = conservativeState.primitive(gas);
  auto consFluxes     = conservativeState.flux(gas, 0);
  auto primFluxes     = primitiveState.flux(gas, 0);

  // P = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))
  // gamma = 1.4
  // E     = 3.0
  // rho   = 1.0
  // u     = 2.0
  // v     = 2.5
  const auto p = (gas.adi() - T(1))
               * (T(3) - T(0.5) * T(1) * (T(2) * T(2) + T(2.5) * T(2.5)));

  // Hand calculations:
  EXPECT_EQ(consFluxes[0], T(1.0) * T(2.0)             ); // rho * u
  EXPECT_EQ(consFluxes[1], T(2.0) * (T(3.0) + p)       ); // u * (E + p)
  EXPECT_EQ(consFluxes[2], T(1.0) * T(2.0) * T(2.0) + p); // rho * u * u + p
  EXPECT_EQ(consFluxes[3], T(1.0) * T(2.0) * T(2.5)    ); // rho * u * v
  EXPECT_EQ(consFluxes[4], T(1.0) * T(2.0) * T(3.0)    ); // rho * u * additional

  EXPECT_EQ(consFluxes[0], primFluxes[0]);
  EXPECT_EQ(consFluxes[1], primFluxes[1]);
  EXPECT_EQ(consFluxes[2], primFluxes[2]);
  EXPECT_EQ(consFluxes[3], primFluxes[3]);
  EXPECT_EQ(consFluxes[4], primFluxes[4]);
}

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