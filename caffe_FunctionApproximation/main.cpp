#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "ANN.h"

using namespace std;

// !!!! TEST_CASE IN COMBINITION WITH REQUIRE IS LIKE A LOOP STRUCTURE !!!!
// !!!! THE TEST_CASE IS EXECUTED FOR EVERY SECTION !!!!

TEST_CASE( "Simple Forward Net scalar input Value -> tanh -> scalar output value", "input_tanh_output.prototxt" ) {
    ANN ann("../caffe_FunctionApproximation/prototxt/input_tanh_output.prototxt");

    SECTION( "single input works" ) {
        float f = -2.0;
        while (f <= 2.0) {
            f += 0.1;
            REQUIRE(ann.forward(f) == Approx(tanh(f)));
        }

    }

}
