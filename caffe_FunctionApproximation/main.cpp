#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "ANN.h"

using namespace std;

// !!!! TEST_CASE IN COMBINITION WITH REQUIRE IS LIKE A LOOP STRUCTURE !!!!
// !!!! THE TEST_CASE IS EXECUTED FOR EVERY SECTION !!!!



TEST_CASE( "Simple Forward Net scalar input Value -> tanh -> scalar output value", "input_tanh_output.prototxt" ) {
    ANN ann("../caffe_FunctionApproximation/prototxt/input_tanh_output.prototxt");

    SECTION( "single input works" ) {
        double d = -2.0;
        while (d <= 2.0) {
            d += 0.1;
            REQUIRE(ann.forward(d) == Approx(tanh(d )));
        }

    }
    SECTION( "vector input works" ) {
        double d = -2.0;
        vector<double> inputValues;
        while (d <= 2.0) {
            d += 0.1;
            inputValues.push_back(d);
        }

        vector<double> outputValues = ann.forward(inputValues);
        REQUIRE(inputValues.size() == outputValues.size());
        for (unsigned int i = 0; i < outputValues.size(); i++) {
            REQUIRE(outputValues[i] == Approx(tanh(inputValues[i])));
        }

    }

}


TEST_CASE( "Simple Forward Net scalar input Value -> innerproduct -> tanh -> innerproduct -> tanh -> scalar output value", "input_tanh_output.prototxt" ) {
    ANN ann("../caffe_FunctionApproximation/prototxt/input__innerproduct_tanh_innerproduct_tanh_output.prototxt");

    SECTION( "single input works" ) {
        double d = -2.0;
        while (d <= 2.0) {
            d += 0.1;
            cout << "output : " << ann.forward(d) << endl;
        }

    }
}
