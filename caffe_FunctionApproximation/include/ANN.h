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


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace std;


class ANN {
    public:
        ANN(const string& modelFile_, const string &trainedFile_ = "");
        void forward (float inputValue_);
        void  setDataOfBLOB(Blob<float>* blobToModify_,int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_, float value_);
        float getDataOfBLOB(Blob<float>* blobToReadFrom_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_);
    private:
        Net<float> *net;

};



#endif // ANN_H
