#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "ANN.h"

using namespace std;

// !!!! TEST_CASE IN COMBINITION WITH REQUIRE IS LIKE A LOOP STRUCTURE !!!!
// !!!! THE TEST_CASE IS EXECUTED FOR EVERY SECTION !!!!

bool nearlyEqual(double val1_, double val2_, double tolerance_) {
    return ( abs (val1_ - val2_) <= tolerance_ );
}

TEST_CASE( "Test nearly equal" ) {
    const double TOLERANCE = 0.05;
    REQUIRE(nearlyEqual(0, 0.05,TOLERANCE));
    REQUIRE(nearlyEqual(0,-0.05,TOLERANCE));
    REQUIRE(nearlyEqual(0, 0.04,TOLERANCE));
    REQUIRE(nearlyEqual(0,-0.04,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual(0, 0.06,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual(0,-0.06,TOLERANCE));
    REQUIRE(nearlyEqual(-0.02,-0.07,TOLERANCE));
    REQUIRE(nearlyEqual(-0.02, 0.03,TOLERANCE));
    REQUIRE(nearlyEqual(-0.02,-0.06,TOLERANCE));
    REQUIRE(nearlyEqual(-0.02, 0.02,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual(-0.02,-0.08,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual(-0.02, 0.04,TOLERANCE));

    REQUIRE(nearlyEqual( 0.05,0,TOLERANCE));
    REQUIRE(nearlyEqual(-0.05,0,TOLERANCE));
    REQUIRE(nearlyEqual( 0.04,0,TOLERANCE));
    REQUIRE(nearlyEqual(-0.04,0,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual( 0.06,0,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual(-0.06,0,TOLERANCE));
    REQUIRE(nearlyEqual(-0.07,-0.02,TOLERANCE));
    REQUIRE(nearlyEqual( 0.03,-0.02,TOLERANCE));
    REQUIRE(nearlyEqual(-0.06,-0.02,TOLERANCE));
    REQUIRE(nearlyEqual( 0.02,-0.02,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual(-0.08,-0.02,TOLERANCE));
    REQUIRE_FALSE(nearlyEqual( 0.04,-0.02,TOLERANCE));
}


TEST_CASE( "Simple Forward Net scalar input Value -> tanh -> scalar output value" ) {
    ANN ann("../caffe_FunctionApproximation/prototxt/very_simple_net.prototxt");


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


TEST_CASE( "Simple Forward Net scalar input Value -> innerproduct -> tanh -> innerproduct -> tanh -> scalar output value") {
    ANN ann("../caffe_FunctionApproximation/prototxt/net_without_loss.prototxt");

    SECTION( "single input works" ) {
        double d = -2.0;
        while (d <= 2.0) {
            d += 0.1;
            // check if output is any random valid tanHyperbolic value
            REQUIRE(ann.forward(d) >= -1);
            REQUIRE(ann.forward(d) <= 1);
        }

    }
}


TEST_CASE("Training ANN") {
    ANN ann("../caffe_FunctionApproximation/prototxt/net_without_loss.prototxt",
            "","../caffe_FunctionApproximation/prototxt/test_solver.prototxt");

    SECTION( "vector learning on random weights works" ) {
        double d = -2.0;
        vector<double> inputValues;
        vector<double> expectedResults;

        while (d <= 2.0) {
            d += 0.1;
            inputValues.push_back(d);
            expectedResults.push_back(tanh(d));
        }

       REQUIRE(ann.train(inputValues,expectedResults));
    }

    SECTION( "vector learning on from-file-loaded weights works" ) {
        double d = -2.0;
        vector<double> inputValues;
        vector<double> expectedResults;

        ann.setTrainedWeightsCaffemodelPath("/home/anon/Desktop/PrivateProjects/Programming/C++/Caffe_Deep_Learning_Framework/Caffe_FunctionApproximation/caffe_FunctionApproximation/caffemodel/train_iter_449980.caffemodel");

        while (d <= 2.0) {
            d += 0.1;
            inputValues.push_back(d);
            expectedResults.push_back(tanh(d));
        }

       REQUIRE(ann.train(inputValues,expectedResults));

       SECTION( "propagate through trained network" ) {
           double d = -2.0;
           vector<double> inputVals;
           vector<double> tanhOut;
           vector<double> annOut;

           while (d <= 2.0) {
               d += 0.1;
               inputVals.push_back(d);
               tanhOut.push_back(tanh(d));
           }

           annOut = ann.forward(inputVals);

           ofstream oFile("test.csv");
           for (int i = 0; i < annOut.size(); i++) {
               oFile << inputVals[i] << "," << tanhOut[i] << "," << annOut[i] << endl;
               cout << "for : " << inputVals[i] << " tanh : "  << tanhOut[i] << endl;
               cout << "for : " << inputVals[i] << " ann  : "  << annOut[i] << endl;
               REQUIRE(nearlyEqual(tanhOut[i],annOut[i],0.05));
           }
           oFile.close();
       }
    }


}
