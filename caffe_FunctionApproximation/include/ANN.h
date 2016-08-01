#ifndef ANN_H
#define ANN_H

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/sgd_solvers.hpp"
#include "caffe/solver.hpp"
#include "google/protobuf/text_format.h"

using namespace caffe;
using namespace std;
//using shared_ptr = std::shared_ptr;


class ANN {
    public:
        /* --- constructors / destructors --- */
        ANN(const string& modelFile_, Phase phase_, const string &trainedFile_ = "");

        /* --- pushing values forward (from input to output) --- */
        double         forward (double         inputValue_);
        vector<double> forward (vector<double> inputValues_);

        /* --- train / optimize weights --- */
        double train (double inputValue_, double expectedResult_, const string &solverFile_);
        double train (vector<double> inputValues_, vector<double> expectedResults_, const string& solverFile_);

        /* --- miscellaneous --- */
        void  setDataOfBLOB(Blob<double>* blobToModify_,int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_, double value_);
        double getDataOfBLOB(Blob<double>* blobToReadFrom_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_);
    private:
        // artificial neural net
        Net<double> *net;
        caffe::shared_ptr<Solver<double> > solver_;

};



#endif // ANN_H
