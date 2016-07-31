#include <iostream>

using namespace std;

#include "ANN.h"

int main() {
    ANN ann("../caffe_FunctionApproximation/prototxt/input_tanh_output.prototxt");
    float f = -2.0;
    while (f <= 2.0) {
        f += 0.1;
        ann.forward(f);
    }

    cout << "Hello World" << endl;
    return 0;
}


