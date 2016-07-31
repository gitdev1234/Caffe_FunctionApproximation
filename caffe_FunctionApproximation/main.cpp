#include <iostream>

using namespace std;

#include "ANN.h"

int main() {
    ANN ann("../caffe_FunctionApproximation/prototxt/test2.prototxt");
    ann.forward(13.4);
    cout << "Hello World" << endl;
    return 0;
}


