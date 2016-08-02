#ifndef ANN_H
#define ANN_H

// STL
#include <vector>
#include <iostream>
#include <fstream>
// caffe
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/solver.hpp"
#include "google/protobuf/text_format.h"

using namespace caffe;
using namespace std;


/**
 * @brief The ANN class - a implementation for simple neural net with one input and one output
 *
 * The ANN class is an implementation of an artificial neural network for solving
 * simple proplems like function approximation which uses the caffe-framework.
 *
 * ANN provides the possibility for training simple artificial neural nets with one
 * input neuron and one output neuron and for calculating an output value by a given input value.
 *
 *
 */
class ANN {
    public:
        /* --- constructors / destructors --- */
        ANN(const string& netStructurePrototxtPath_, const string &trainedWeightsCaffemodelPath_ = "",
            const string& solverParametersPrototxtPath_ = "");

        /* --- getter / setter --- */
        string getNetStructurePrototxtPath     () const {return netStructurePrototxtPath     ;};
        string getTrainedWeightsCaffemodelPath () const {return trainedWeightsCaffemodelPath ;};
        string getSolverParametersPrototxtPath () const {return solverParametersPrototxtPath ;};

        void setNetStructurePrototxtPath     (const string& val_) {netStructurePrototxtPath     = val_;};
        void setTrainedWeightsCaffemodelPath (const string& val_) {trainedWeightsCaffemodelPath = val_;};
        void setSolverParametersPrototxtPath (const string& val_) {solverParametersPrototxtPath = val_;};

        /* --- pushing values forward (from input to output) --- */
        double         forward (double         inputValue_);
        vector<double> forward (vector<double> inputValues_);

        /* --- train / optimize weights --- */
        bool train (vector<double> inputValues_, vector<double> expectedOutputValues_);

     private:
        // artificial neural net
        Net<double> *net;
        caffe::shared_ptr<Solver<double> > solver_;
        // paths of important files
        string netStructurePrototxtPath;
        string trainedWeightsCaffemodelPath;
        string solverParametersPrototxtPath;


        /* --- miscellaneous --- */
        void  setDataOfBLOB(Blob<double>* blobToModify_,int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_, double value_);
        double getDataOfBLOB(Blob<double>* blobToReadFrom_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_);


};



#endif // ANN_H
