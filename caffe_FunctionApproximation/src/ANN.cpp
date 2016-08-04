#include "ANN.h"

/* --- constructors / destructors --- */

/**
 * @brief ANN::ANN constructor of class ANN
 *
 * @param netStructurePrototxtPath_     path of prototxt-file which describes the net structure
 * @param trainedWeightsCaffemodelPath_ path of caffemodel-file which contains the trained weights of the net
 * @param solverParametersPrototxtPath_ path of prototxt-file which describes the solver-parameters and the links to the net which is to optimize
 *
 * constructor of class ANN
 *  1. sets processing mode (CPU / GPU) depending on previous define CPU_ONLY
 *  2. sets the given paths in private attributes
 */
ANN::ANN(const string& netStructurePrototxtPath_, const string& trainedWeightsCaffemodelPath_, const string &solverParametersPrototxtPath_) {
    // set processing source
    #ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
    #else
      Caffe::set_mode(Caffe::GPU);
    #endif

    // set paths
    setNetStructurePrototxtPath(netStructurePrototxtPath_);
    setTrainedWeightsCaffemodelPath(trainedWeightsCaffemodelPath_);
    setSolverParametersPrototxtPath(solverParametersPrototxtPath_);

}

/* --- pushing values forward (from input to output) --- */

/**
 * @brief ANN::forward propagates a scalar double value through the net
 * @param inputValue_ value which is to propagate through the net
 * @return returns the scalar output value of the net
 *
 * NOTICE : This is to use for nets with only one input-neuron and one output-neuron
 */
double ANN::forward(double inputValue_) {

    // load network-structure from prototxt-file
    net = new Net<double>(getNetStructurePrototxtPath(),caffe::TEST);

    // load weights
    string trainedWeightsCaffemodelPath_l = getTrainedWeightsCaffemodelPath();
    if (trainedWeightsCaffemodelPath_l != "") {
        net->CopyTrainedLayersFrom(trainedWeightsCaffemodelPath_l);
    }

    // create BLOB for input layer - data
    Blob<double>* inputLayer = net->input_blobs()[0];

    // set dimesions of input layer
    // --> for normal caffe works with images, therefore the data
    // --> typically is 4 dimensional
    // --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
    // --> in this case we use 1-dimensional data, therefore the data-dimension is 1*1*1*1
    int num      = 1;
    int channels = 1;
    int height   = 1;
    int width    = 1;
    vector<int> dimensionsOfInputData = {num,channels,height,width};
    inputLayer->Reshape(dimensionsOfInputData);

    // forward dimension-change to all layers.
    net->Reshape();

    // insert inputValue into inputLayer
    setDataOfBLOB(inputLayer,0,0,0,0,inputValue_);

    // propagate inputValue through layers
    net->Forward();

    // create BLOB for outputLayer
    Blob<double>* outputLayer = net->output_blobs()[0];

    // return the only value in output Layer
    return getDataOfBLOB(outputLayer,0,0,0,0);
}

/**
 * @brief ANN::forward propagates a vector of double values through the net
 * @param inputValues_ vector of double values which are to propagate through the net
 * @return returns the vector of output values of the net
 *
 * NOTICE : This is to use for nets with one to many input-neurons and one to many output-neurons
 */
vector<double> ANN::forward(vector<double> inputValues_) {

    // load network-structure from prototxt-file
    net = new Net<double>(getNetStructurePrototxtPath(),caffe::TEST);

    // load weights
    string trainedWeightsCaffemodelPath_l = getTrainedWeightsCaffemodelPath();
    if (trainedWeightsCaffemodelPath_l != "") {
        trainedWeightsCaffemodelPath_l = "/home/anon/Desktop/PrivateProjects/Programming/C++/Caffe_Deep_Learning_Framework/Caffe_FunctionApproximation/build-caffe_FunctionApproximation-Unnamed-Debug/train_iter_450000.caffemodel";
        net->CopyTrainedLayersFrom(trainedWeightsCaffemodelPath_l);
    }

    // create BLOB for input layer
    Blob<double>* inputLayer = net->input_blobs()[0];

    // set dimesions of input layer
    // --> for normal caffe works with images, therefore the data
    // --> typically is 4 dimensional
    // --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
    // --> in this case we use 1-dimensional data, therefore the data-dimension is 1*1*1*1
    int num      = inputValues_.size();
    int channels = 1;
    int height   = 1;
    int width    = 1;
    vector<int> dimensionsOfInputData = {num,channels,height,width};
    inputLayer->Reshape(dimensionsOfInputData);

    // forward dimension change to all layers.
    net->Reshape();

    // insert inputValue into inputLayer
    for (unsigned int i = 0; i < inputValues_.size(); i++) {
        setDataOfBLOB(inputLayer,i,0,0,0,inputValues_[i]);
    }

    // propagate inputValue through layers
    net->Forward();

    // create BLOB for outputLayer
    Blob<double>* outputLayer = net->output_blobs()[0];
    cout << "num : " << outputLayer->num() << endl;
    cout << "channels : " << outputLayer->channels() << endl;
    cout << "height : " << outputLayer->height() << endl;
    cout << "width : " << outputLayer->width() << endl;


    // copy values in output Layer to 1-dimensional-vector of values
    vector<double> result;
    for (int i = 0; i < outputLayer->num(); i++) {
        result.push_back(getDataOfBLOB(outputLayer,i,0,0,0));
    }

    // return vector of values
    return result;
}

vector<vector<double> > ANN::forward(vector<vector<double> > inputValues_) {

    // load network-structure from prototxt-file
    net = new Net<double>(getNetStructurePrototxtPath(),caffe::TEST);

    // load weights
    string trainedWeightsCaffemodelPath_l = getTrainedWeightsCaffemodelPath();
    if (trainedWeightsCaffemodelPath_l != "") { // TODO
        trainedWeightsCaffemodelPath_l = "/home/anon/Desktop/PrivateProjects/Programming/C++/Caffe_Deep_Learning_Framework/Caffe_FunctionApproximation/build-caffe_FunctionApproximation-Unnamed-Debug/train_iter_450000.caffemodel";
        net->CopyTrainedLayersFrom(trainedWeightsCaffemodelPath_l);
    }

    // create BLOB for input layer
    Blob<double>* inputLayer = net->input_blobs()[0];

    // set dimesions of input layer
    // --> for normal caffe works with images, therefore the data
    // --> typically is 4 dimensional
    // --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
    // --> in this case we use 1-dimensional data, therefore the data-dimension is 1*1*1*1
    int num      = inputValues_.size();
    int channels = inputValues_[0].size();
    int height   = 1;
    int width    = 1;
    vector<int> dimensionsOfInputData = {num,channels,height,width};
    inputLayer->Reshape(dimensionsOfInputData);

    // forward dimension change to all layers.
    net->Reshape();

    // insert inputValue into inputLayer

    // iterate all neurons for every inputData-dataset
    for (unsigned int inputDataSetIndex = 0; inputDataSetIndex < inputValues_.size(); inputDataSetIndex++) {
        for (unsigned int inputNeuronIndex = 0; inputNeuronIndex < inputValues_[inputDataSetIndex].size(); inputNeuronIndex++) {
            setDataOfBLOB(inputLayer,inputDataSetIndex,inputNeuronIndex,0,0,inputValues_[inputDataSetIndex][inputNeuronIndex]);
        }
    }


    // propagate inputValue through layers
    net->Forward();

    // create BLOB for outputLayer
    Blob<double>* outputLayer = net->output_blobs()[0];
    cout << "num : " << outputLayer->num() << endl;
    cout << "channels : " << outputLayer->channels() << endl;
    cout << "height : " << outputLayer->height() << endl;
    cout << "width : " << outputLayer->width() << endl;


    // copy values in output Layer to 1-dimensional-vector of values
    vector<vector<double>> result (outputLayer->num(), std::vector<double>(outputLayer->channels()));
    for (int outputDataSetIndex = 0; outputDataSetIndex < outputLayer->num(); outputDataSetIndex++) {
        for(int outputNeuronIndex = 0; outputNeuronIndex < outputLayer->channels(); outputNeuronIndex++) {
            result[outputDataSetIndex][outputNeuronIndex] =getDataOfBLOB(outputLayer,outputDataSetIndex,outputNeuronIndex,0,0);
        }
    }

    // return vector of values
    return result;
}

/* --- train / optimize weights --- */

/**
 * @brief ANN::train trains the network with the given inputs and expected outputs
 * @param inputValues_ vector of inputValues
 * @param expectedOutputValues_ vector of output values
 * @return returns true if training has succesfully ended, otherwise false
 *
 * The train function executes a learning to the net by doing the following steps :
 *   1. create a solver object which encapsulate and controls the net solverParametersPrototxtPath-file
 *   2. load the input data and the expected output data to the net
 *   3. execute solver_->Solve, which
 *        3.1 automatically loops through the input data
 *        3.2 propagates the input data through the net,
 *        3.3 calculates the current loss of the output
 *        3.4 calculates deltas for every weight, depending on the loss
 *        3.5 calculates new weights
 *        3.6 outputs preliminary results and the final result of the trained weights
 *            into *.caffemodel - files
 *
 * NOTICE : the size of inputValues and expected output values has to be equal
 *          otherwise the function stops and returns false
 * NOTICE : the file, which is located at getSolverParametersPrototxtPath has to be a valid
 *          google-protobuf file which can be used to specify a caffe-solver, otherwise the
 *          function stops and returns true
 *
 */
bool ANN::train (vector<double> inputValues_, vector<double> expectedOutputValues_) {
    if (inputValues_.size() != expectedOutputValues_.size()) {
        cout << "Error : inputValues_ and expectedOutputValues_ have different lengths" << endl;
        return false;
    } else {
        SolverParameter param;
        switch (Caffe::mode()) {
          case Caffe::CPU:
            param.set_solver_mode(SolverParameter_SolverMode_CPU);
            break;
          case Caffe::GPU:
            param.set_solver_mode(SolverParameter_SolverMode_GPU);
            break;
          default:
            LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
        }

        // --- read solver parameters from file ---

        // read file into string
        std::ifstream iFile;
        iFile.open(getSolverParametersPrototxtPath());
        stringstream sstr;
        sstr << iFile.rdbuf();
        string str;
        str = sstr.str();

        // create solver parameter by string
        if (!google::protobuf::TextFormat::ParseFromString(str, &param)) {
            cout << "Error : solver prototxt file is not valid" << endl;
            return false;
        } else {

            // create solver by parameter
            solver_.reset(new SGDSolver<double>(param));

            // load weights
            string trainedWeightsCaffemodelPath_l = getTrainedWeightsCaffemodelPath();
            if (trainedWeightsCaffemodelPath_l != "") {
                solver_->net()->CopyTrainedLayersFrom(trainedWeightsCaffemodelPath_l);
            }

            // --- load input data and expected output data into solver_->net ---

            // create BLOB for inputlayer - input data
            Blob<double>* inputDataBLOB = solver_->net()->input_blobs()[0];

            // create BLOB for inputlayer - expected output data
            Blob<double>* expectedOutputDataBLOB = solver_->net()->input_blobs()[1];

            // set dimesions of input layer
            // --> for normal caffe works with images, therefore the data
            // --> typically is 4 dimensional
            // --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
            // --> in this case we use 1-dimensional data, therefore the data-dimension is numberOfInputValues*1*1*1
            int num      = inputValues_.size();
            int channels = 1;
            int height   = 1;
            int width    = 1;
            vector<int> dimensionsOfData = {num,channels,height,width};

            // set dimensions of input data
            inputDataBLOB->Reshape(dimensionsOfData);

            // set dimensions of expected output data
            expectedOutputDataBLOB->Reshape(dimensionsOfData);

            // forward dimension-change to all layers
            solver_->net()->Reshape();

            // insert input values and expected output values into
            // BLOBs of input layer and output layer
            for (unsigned int i = 0; i < inputValues_.size(); i++) {
                setDataOfBLOB(inputDataBLOB,i,0,0,0,inputValues_[i]);
                setDataOfBLOB(expectedOutputDataBLOB,i,0,0,0,expectedOutputValues_[i]);
            }

            // start training
            //  --> this is done by Solve(), which automatically does the following steps
            //      --> 1. propagates input data through solver_->net
            //             by automatically using solver_->net()->forward
            //             as the  output layer is a loss-layer the forward()
            //             produces a loss value (the badness of the current weights)
            //      --> 2. by knowing the loss and using solver_->net()->backward() it automatically
            //             calculates a gradient for every weight (every connection) in the net
            //             (gradient == a delta of how much the weights have to get changed)
            //      --> 3. by knowing the gradients it automatically calculates the new weights
            //  --> these three steps are executed for every input-value and for every learning iteration step
            //  --> during Solve() preliminary results for the weights are saved to the directory defined in
            //      solverFile_
            //  --> the frequency of creating preliminary results as well as the number of training iterations
            //      and other parameters are defined in solverFile_
            solver_->Solve();

            // save current trained weights
            stringstream tempPath;
            tempPath << param.snapshot_prefix() << "_iter_" << param.max_iter() << ".caffemodel";
            setTrainedWeightsCaffemodelPath(tempPath.str());
            solver_->Snapshot();
            return true;
        }
    }
}

bool ANN::train(vector<vector<double> > inputValues_, vector<double> expectedOutputValues_) {
    if (inputValues_.size() != expectedOutputValues_.size()) {
        cout << "Error : inputValues_ and expectedOutputValues_ have different lengths" << endl;
        return false;
    } else {
        SolverParameter param;
        switch (Caffe::mode()) {
          case Caffe::CPU:
            param.set_solver_mode(SolverParameter_SolverMode_CPU);
            break;
          case Caffe::GPU:
            param.set_solver_mode(SolverParameter_SolverMode_GPU);
            break;
          default:
            LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
        }

        // --- read solver parameters from file ---

        // read file into string
        std::ifstream iFile;
        iFile.open(getSolverParametersPrototxtPath());
        stringstream sstr;
        sstr << iFile.rdbuf();
        string str;
        str = sstr.str();
        cout << str << endl;

        // create solver parameter by string
        if (!google::protobuf::TextFormat::ParseFromString(str, &param)) {
            cout << "Error : solver prototxt file is not valid" << endl;
            return false;
        } else {

            // create solver by parameter
            solver_.reset(new SGDSolver<double>(param));

            // load weights
            string trainedWeightsCaffemodelPath_l = getTrainedWeightsCaffemodelPath();
            if (trainedWeightsCaffemodelPath_l != "") {
                solver_->net()->CopyTrainedLayersFrom(trainedWeightsCaffemodelPath_l);
            }

            // --- load input data and expected output data into solver_->net ---

            // create BLOB for inputlayer - input data
            Blob<double>* inputDataBLOB = solver_->net()->input_blobs()[0];

            // create BLOB for inputlayer - expected output data
            Blob<double>* expectedOutputDataBLOB = solver_->net()->input_blobs()[1];


            // set dimesions of input layer
            // --> for normal caffe works with images, therefore the data
            // --> typically is 4 dimensional
            // --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
            // --> in this case we use 1-dimensional data, therefore the data-dimension is numberOfInputValues*1*1*1
            int num      = inputValues_.size();
            int channels = inputValues_[0].size();
            int height   = 1;
            int width    = 1;
            vector<int> dimensionsOfData = {num,channels,height,width};

            // set dimensions of input data
            inputDataBLOB->Reshape(dimensionsOfData);

            // set dimensions of expected output data
            expectedOutputDataBLOB->Reshape(dimensionsOfData);

            // forward dimension-change to all layers
            solver_->net()->Reshape();

            // insert input values for all neurons for every inputData-dataset
            // into BLOB of input layer

            for (unsigned int inputDataSetIndex = 0; inputDataSetIndex < inputValues_.size(); inputDataSetIndex++) {
                for (unsigned int inputNeuronIndex = 0; inputNeuronIndex < inputValues_[inputDataSetIndex].size(); inputNeuronIndex++) {
                    setDataOfBLOB(inputDataBLOB,inputDataSetIndex,inputNeuronIndex,0,0,inputValues_[inputDataSetIndex][inputNeuronIndex]);
                }
            }

            // insert expected output values into
            // BLOB of output layer
            for (unsigned int i = 0; i < inputValues_.size(); i++) {
                setDataOfBLOB(expectedOutputDataBLOB,i,0,0,0,expectedOutputValues_[i]);
            }

            // start training
            //  --> this is done by Solve(), which automatically does the following steps
            //      --> 1. propagates input data through solver_->net
            //             by automatically using solver_->net()->forward
            //             as the  output layer is a loss-layer the forward()
            //             produces a loss value (the badness of the current weights)
            //      --> 2. by knowing the loss and using solver_->net()->backward() it automatically
            //             calculates a gradient for every weight (every connection) in the net
            //             (gradient == a delta of how much the weights have to get changed)
            //      --> 3. by knowing the gradients it automatically calculates the new weights
            //  --> these three steps are executed for every input-value and for every learning iteration step
            //  --> during Solve() preliminary results for the weights are saved to the directory defined in
            //      solverFile_
            //  --> the frequency of creating preliminary results as well as the number of training iterations
            //      and other parameters are defined in solverFile_
            solver_->Solve();

            // save current trained weights
            stringstream tempPath;
            tempPath << param.snapshot_prefix() << "_iter_" << param.max_iter() << ".caffemodel";
            setTrainedWeightsCaffemodelPath(tempPath.str());
            solver_->Snapshot();
            return true;
        }
    }
}



/* --- miscellaneous --- */

vector<double> ANN::zTransformVector(const vector<double>& vectorToTransform_) {

    vector<double> result = vectorToTransform_;

    // calculate mean
    double sum = std::accumulate(vectorToTransform_.begin(), vectorToTransform_.end(), 0.0);
    double mean = double(sum) / double(vectorToTransform_.size());
    // calculate standard deviation
    double sq_sum = std::inner_product(vectorToTransform_.begin(), vectorToTransform_.end(), vectorToTransform_.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / double(vectorToTransform_.size()-1) - mean * mean);

    for (int i = 0 ; i < vectorToTransform_.size(); i++) {
        result[i] = (double(vectorToTransform_[i]) - double(mean)) / double(stdev);
    }

    return result;

}

vector<double> ANN::reZTransformVector(const vector<double> &vectorToReTransform_, const vector<double> &vectorBeforeZTransform_) {
    vector<double> result = vectorToReTransform_;

    // calculate mean
    double sum = std::accumulate(vectorBeforeZTransform_.begin(), vectorBeforeZTransform_.end(), 0.0);
    double mean = double(sum) / double(vectorBeforeZTransform_.size());
    // calculate standard deviation
    double sq_sum = std::inner_product(vectorBeforeZTransform_.begin(), vectorBeforeZTransform_.end(), vectorBeforeZTransform_.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / double(vectorBeforeZTransform_.size()-1) - mean * mean);

    for (int i = 0 ; i < vectorToReTransform_.size(); i++) {
        result[i] = double(vectorToReTransform_[i]) * double(stdev) + double(mean);
    }

    return result;
}

vector<double> ANN::scaleVector(const vector<double> &vectorToScale_, double scaleFactor_, bool minimize_) {
    vector<double> result = vectorToScale_;

    for (int i = 0 ; i < vectorToScale_.size(); i++) {
        if (minimize_) {
            result[i] = double(vectorToScale_[i]) / double(scaleFactor_);
        } else {
            result[i] = double(vectorToScale_[i]) * double(scaleFactor_);
        }
    }

    return result;
}

vector<vector<double> > ANN::scaleVector(const vector<vector<double> > &vectorToScale_, double scaleFactor_, bool minimize_) {
    vector< vector<double> > result = vectorToScale_;

    for (int i = 0 ; i < vectorToScale_.size(); i++) {
        result[i] = scaleVector(vectorToScale_[i],scaleFactor_,minimize_);
    }

    return result;
}



/**
 * @brief ANN::setDataOfBLOB sets the data at the given indexes within the blobToModify_ to value_
 * @param blobToModify_ the blob which is to modify
 * @param indexNum_     the index of the first dimension (index of image) valid indexes are from zero to blobToModify_->num() - 1
 * @param indexChannel_ the index of the second dimension (index of channel) valid indexes are from zero to blobToModify_->channels() - 1
 * @param indexHeight_  the index of the third dimension (y-index of pixel) valid indexes are from zero to blobToModify_->height() - 1
 * @param indexWidth_   the index of the fourth dimension (x-index of pixel) valid indexes are from zero to blobToModify_->width - 1
 * @param value_        the new value for data at the given indexes within the blobToModify_
 *
 * for normal caffe works with images, therefore the data typically is 4 dimensional
 * --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
 *
 * To use blobs in whatever way (e.g. normal data which is not images) it might
 * be useful to set the content of the blob manually.
 * This function provides the functionality of doing this.
 *
 * NOTICE : If an invalid index is handed into the function, the function just does nothing, except for
 *          printing an error to the console.
 *
 * NOTICE : ALL INDEXES USED FOR ACCESSING THE BLOB ARE ZERO-BASED
 *
 */
void ANN::setDataOfBLOB(Blob<double> *blobToModify_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_, double value_) {
    // check if index is invalid
    if ( (indexNum_     < 0) || (indexNum_     > blobToModify_->num()      - 1) ||
         (indexChannel_ < 0) || (indexChannel_ > blobToModify_->channels() - 1) ||
         (indexHeight_  < 0) || (indexHeight_  > blobToModify_->height()   - 1) ||
         (indexWidth_   < 0) || (indexWidth_   > blobToModify_->width()    - 1) ){
        cout << "Error : please use valid indexes!" << endl;
    } else {
        // create a pointer, that points to the first value inside the blobToModify
        double* pointerToBlobValue = blobToModify_->mutable_cpu_data();

        // calculate the address of the requested indexes
        int addressIncrement  = indexNum_     * blobToModify_->channels() * blobToModify_->height() * blobToModify_->width();
            addressIncrement += indexChannel_ * blobToModify_->height()   * blobToModify_->width();
            addressIncrement += indexHeight_  * blobToModify_->width();
            addressIncrement += indexWidth_;

         // let the pointer point to the requested address
         pointerToBlobValue += addressIncrement;

         // set the value at the requested request
         *pointerToBlobValue = value_;
    }
}

/**
 * @brief ANN::getDataOfBLOB reads the data stored at the given index within blobToReadFrom_
 * @param blobToReadFrom_ the blob to read from
 * @param indexNum_     the index of the first dimension (index of image) valid indexes are from zero to blobToModify_->num() - 1
 * @param indexChannel_ the index of the second dimension (index of channel) valid indexes are from zero to blobToModify_->channels() - 1
 * @param indexHeight_  the index of the third dimension (y-index of pixel) valid indexes are from zero to blobToModify_->height() - 1
 * @param indexWidth_   the index of the fourth dimension (x-index of pixel) valid indexes are from zero to blobToModify_->width - 1
 * @return returns the data stored at the given index within blobToReadFrom_
 *
 * for normal caffe works with images, therefore the data typically is 4 dimensional
 * --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
 *
 * To use blobs in whatever way (e.g. normal data which is not images) it might
 * be useful to get the content of the blob manually.
 * This function provides the functionality of doing this.
 *
 * NOTICE : If an invalid index is handed into the function, the function returns zero and
 *          prints an error to console
 *
 * NOTICE : ALL INDEXES USED FOR ACCESSING THE BLOB ARE ZERO-BASED
 *
 */
double ANN::getDataOfBLOB(Blob<double> *blobToReadFrom_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_) {
    // check if index is invalid
    if ( (indexNum_     < 0) || (indexNum_     > blobToReadFrom_->num()      - 1) ||
         (indexChannel_ < 0) || (indexChannel_ > blobToReadFrom_->channels() - 1) ||
         (indexHeight_  < 0) || (indexHeight_  > blobToReadFrom_->height()   - 1) ||
         (indexWidth_   < 0) || (indexWidth_   > blobToReadFrom_->width()    - 1) ){
        cout << "Error : please use valid indexes!" << endl;
        return 0;
    } else {
        return blobToReadFrom_->data_at(indexNum_,indexChannel_,indexHeight_,indexWidth_);
    }
}
