#include <gtest/gtest.h>
#include "SystemParameters.h"
#include "InvertedPendulumLinearDynamics.h"

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

TEST(SystemParametersTest, ReadFile){
    auto sysParams = SystemParameters("parametersTest.xml");
}

TEST(SystemParametersTest, ParametersLoaded){
    auto sysParams = SystemParameters("parametersTest.xml");
    EXPECT_EQ(sysParams.params.cart.mass, 0.5);
    EXPECT_EQ(sysParams.params.gravity, 9.8);
}

TEST(SystemParametersTest, MissingCartMass){
    auto sysParams = SystemParameters("parametersTestMissingCartMass.xml");
    EXPECT_EQ(sysParams.params.cart.mass, 0);
}

TEST(SystemParametersTest, MissingFriction){
    auto sysParams = SystemParameters("parametersTestMissingFriction.xml");
    EXPECT_EQ(sysParams.params.friction, 0);
}

TEST(SystemParametersTest, FrictionEmpty){
    auto sysParams = SystemParameters("parametersTestFrictionEmpty.xml");
    EXPECT_EQ(sysParams.params.friction, 0);
}

TEST(InvertedPendulumLinearDynamicsTest, CreationTest){
    auto invPendLinDyn = InvertedPendulumLinearDynamics(SystemParameters("parametersTest.xml"));
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}