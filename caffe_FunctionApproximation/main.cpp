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


TEST_CASE ("z-transformation") {
    ANN ann("");
    vector<double> randVals = {
        3.5608959524,  0.2071982273,  5.0710065779, -9.0429632133, -2.9011220089, 5.9242935898,
        8.1078151753, -7.9773012735, -1.0035576625,  2.9495603475, -5.984263923,  1.5644301055,
        8.113492392,  -5.0453927834, -0.6696311967, -2.6061569643, -0.1676226198, 0.7153041032,
        5.341395319,  -4.9502959987, -0.7041797996,  6.2834974588,  2.6638055057, 8.5452458146,
        7.5833344297,  8.7057468691, -6.4355135593, -1.6960091889, -4.4790856098, 3.9688208792
    };
    vector<double> transformedVals = ann.zTransformVector(randVals);

    double tolerance = 0.001;
    REQUIRE(nearlyEqual(transformedVals[0] , 0.51371118,tolerance));
    REQUIRE(nearlyEqual(transformedVals[1] ,-0.12292753,tolerance));
    REQUIRE(nearlyEqual(transformedVals[2] , 0.80037830,tolerance));
    REQUIRE(nearlyEqual(transformedVals[3] ,-1.87890295,tolerance));
    REQUIRE(nearlyEqual(transformedVals[4] ,-0.71298576,tolerance));
    REQUIRE(nearlyEqual(transformedVals[5] , 0.96235937,tolerance));
    REQUIRE(nearlyEqual(transformedVals[6] , 1.37686135,tolerance));
    REQUIRE(nearlyEqual(transformedVals[7] ,-1.67660635,tolerance));
    REQUIRE(nearlyEqual(transformedVals[8] ,-0.35276758,tolerance));
    REQUIRE(nearlyEqual(transformedVals[9] , 0.39766020,tolerance));
    REQUIRE(nearlyEqual(transformedVals[10] ,-1.29826435,tolerance));
    REQUIRE(nearlyEqual(transformedVals[11] , 0.13471834,tolerance));
    REQUIRE(nearlyEqual(transformedVals[12] , 1.37793906,tolerance));
    REQUIRE(nearlyEqual(transformedVals[13] ,-1.12003669,tolerance));
    REQUIRE(nearlyEqual(transformedVals[14] ,-0.28937769,tolerance));
    REQUIRE(nearlyEqual(transformedVals[15] ,-0.65699200,tolerance));
    REQUIRE(nearlyEqual(transformedVals[16] ,-0.19408047,tolerance));
    REQUIRE(nearlyEqual(transformedVals[17] ,-0.02647284,tolerance));
    REQUIRE(nearlyEqual(transformedVals[18] , 0.85170670,tolerance));
    REQUIRE(nearlyEqual(transformedVals[19] ,-1.10198429,tolerance));
    REQUIRE(nearlyEqual(transformedVals[20] ,-0.29593612,tolerance));
    REQUIRE(nearlyEqual(transformedVals[21] , 1.03054771,tolerance));
    REQUIRE(nearlyEqual(transformedVals[22] , 0.34341482,tolerance));
    REQUIRE(nearlyEqual(transformedVals[23] , 1.45989962,tolerance));
    REQUIRE(nearlyEqual(transformedVals[24] , 1.27729819,tolerance));
    REQUIRE(nearlyEqual(transformedVals[25] , 1.49036784,tolerance));
    REQUIRE(nearlyEqual(transformedVals[26] ,-1.38392591,tolerance));
    REQUIRE(nearlyEqual(transformedVals[27] ,-0.48421694,tolerance));
    REQUIRE(nearlyEqual(transformedVals[28] ,-1.01253354,tolerance));
    REQUIRE(nearlyEqual(transformedVals[29] , 0.59114833,tolerance));

    vector<double> reTransformedVals = ann.reZTransformVector(transformedVals,randVals);
    REQUIRE(reTransformedVals.size() == transformedVals.size());
    REQUIRE(transformedVals.size()   == randVals.size());
    for (int i = 0; i < reTransformedVals.size(); i++) {
        REQUIRE(nearlyEqual(reTransformedVals[i],randVals[i],0.000001));
    }


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


TEST_CASE("Training ANN for tanh") {
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

TEST_CASE("Training ANN for x^2 + x") {
    ANN ann("../caffe_FunctionApproximation/prototxt/net_without_loss.prototxt",
            "","../caffe_FunctionApproximation/prototxt/test_solver.prototxt");

    SECTION( "vector learning on random weights works" ) {
        double d = -2.0;
        vector<double> inputValues;
        vector<double> expectedResults;

        while (d <= 2.0) {
            d += 0.1;
            inputValues.push_back(d);
            expectedResults.push_back(d*d+d);
        }

       REQUIRE(ann.train(inputValues,expectedResults));

       SECTION( "propagate through trained network" ) {
           vector<double> annOut;
           annOut = ann.forward(inputValues);

           ofstream oFile("x_square_plus_x.csv");
           for (int i = 0; i < annOut.size(); i++) {
               oFile << inputValues[i] << "," << expectedResults[i] << "," << annOut[i] << endl;
               cout << "for : " << inputValues[i] << " x^2+x : "  << expectedResults[i] << endl;
               cout << "for : " << inputValues[i] << " ann   : "  << annOut[i] << endl;
               REQUIRE(nearlyEqual(expectedResults[i],annOut[i],0.05));
           }
           oFile.close();
       }
    }
}
