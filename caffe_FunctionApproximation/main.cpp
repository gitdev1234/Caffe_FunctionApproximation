#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "ANN.h"

using namespace std;

// !!!! TEST_CASE IN COMBINITION WITH REQUIRE IS LIKE A LOOP STRUCTURE !!!!
// !!!! THE TEST_CASE IS EXECUTED FOR EVERY SECTION !!!!



TEST_CASE( "Simple Forward Net scalar input Value -> tanh -> scalar output value" ) {
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


TEST_CASE( "Simple Forward Net scalar input Value -> innerproduct -> tanh -> innerproduct -> tanh -> scalar output value") {
    ANN ann("../caffe_FunctionApproximation/prototxt/input__innerproduct_tanh_innerproduct_tanh_output.prototxt");

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
    ANN ann("../caffe_FunctionApproximation/prototxt/input__innerproduct_tanh_innerproduct_tanh_output_loss.prototxt",
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
           ann.setNetStructurePrototxtPath("../caffe_FunctionApproximation/prototxt/input__innerproduct_tanh_innerproduct_tanh_output.prototxt");
           while (d <= 2.0) {
               d += 0.1;
               inputVals.push_back(d);
               tanhOut.push_back(tanh(d));
               //annOut.push_back(ann.forward(d));
           }

           //ann.setNetStructurePrototxtPath();
           annOut = ann.forward(inputVals);

           ofstream oFile("test.csv");
           for (int i = 0; i < annOut.size(); i++) {
               oFile << inputVals[i] << "," << tanhOut[i] << "," << annOut[i] << endl;
               cout << "for : " << inputVals[i] << " tanh : "  << tanhOut[i] << endl;
               cout << "for : " << inputVals[i] << " ann  : "  << annOut[i] << endl;
           }
           oFile.close();
       }
    }


}

