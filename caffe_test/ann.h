#ifndef ANN_H
#define ANN_H

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"

using namespace caffe;
using namespace std;


class ANN {
    public:
        ANN(const string& model_file);

};

ANN::ANN(const string& model_file) {
    // set processing source
    #ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
    #else
      Caffe::set_mode(Caffe::GPU);
    #endif

    // load network-structure from prototxt-file
    Net<float> net(model_file,caffe::TEST);
}

#endif // ANN_H
