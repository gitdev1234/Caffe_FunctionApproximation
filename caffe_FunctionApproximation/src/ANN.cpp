#include "ANN.h"

/* --- constructors / destructors --- */

/**
 * @brief ANN::ANN constructor of class ANN
 * @param modelFile_ path of prototxt-file which cotains description of net-structure
 * @param trainedFile_ path of caffemodel-file which contains already trained weights
 *
 * constructor of class ANN
 *  1. sets processing mode (CPU / GPU) depending on previous define CPU_ONLY
 *  2. loads net-structure from prototxt-file at path modelFile_
 *  3. loads trained weights from caffemodel-file at path trainedFile_
 */
ANN::ANN(const string& modelFile_, caffe::Phase phase_, const string& trainedFile_) {
    // set processing source
    #ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
    #else
      Caffe::set_mode(Caffe::GPU);
    #endif

    // load network-structure from prototxt-file
    net = new Net<double>(modelFile_,phase_);

    // load weights
    if (trainedFile_ != "") {
        net->CopyTrainedLayersFrom(trainedFile_);
    }
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
 * @param inputValue_ vector of double values which are to propagate through the net
 * @return returns the vector of output values of the net
 *
 * NOTICE : This is to use for nets with one to many input-neurons and one to many output-neurons
 */
vector<double> ANN::forward(vector<double> inputValues_) {

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

/* --- train / optimize weights --- */
double ANN::train (double inputValue_, double expectedResult_, const string& solverFile_) {


    /*
    // propagate inputValue through layers
    net->Forward();
    vector<string> names = net->blob_names();
    cout << "names : ";
    for (int i = 0; i< names.size(); i++) {
        cout << names[i] << ",";
    }
    cout << endl;
    caffe::shared_ptr<Blob<double> > temp = net->blob_by_name("activatedOutputLayer");
    const double *begin = temp->cpu_data();
    const double *end = begin + temp->channels();
    vector<double> vec = vector<double>(begin, end);
    cout << vec[0];


   // cout << "atte" << &temp << endl;
    //const double* probs_out = temp->cpu_data();
    //int num_2 = (*temp).num();
    //int chann = temp->channels();
    //int heigh = temp->height();
    //int widt = temp->width();
*/

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
    //caffe::shared_ptr<Solver<double> > solver_;


    std::ifstream iFile;
    iFile.open(solverFile_);
    stringstream sstr;
    sstr << iFile.rdbuf();
    string str;
    str = sstr.str();

    google::protobuf::TextFormat::ParseFromString(str, &param);

    solver_.reset(new SGDSolver<double>(param));
    solver_->net()->input_blobs();


    // ------------

    // create BLOB for inputlayer - input data
    Blob<double>* inputLayer = solver_->net()->input_blobs()[0];

    // create BLOB for inputlayer - expected output data
    Blob<double>* label      = solver_->net()->input_blobs()[1];

    // set dimesions of inputlayer - input data
    // --> for normal caffe works with images, therefore the data
    // --> typically is 4 dimensional
    // --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
    // --> in this case we use 1-dimensional data, therefore the data-dimension is 1*1*1*1
    int num      = 1;
    int channels = 1;
    int height   = 1;
    int width    = 1;
    vector<int> dimensionsOfInputLayer = {num,channels,height,width};
    inputLayer->Reshape(dimensionsOfInputLayer);

    // set dimensions of inputlayer - expected output data
    label->Reshape(dimensionsOfInputLayer);

    // forward dimension-change to all layers.
    solver_->net()->Reshape();

    // insert inputValue into inputLayer - input data
    setDataOfBLOB(inputLayer,0,0,0,0,inputValue_);

    // insert expected output value into inputLayer - expected output data
    setDataOfBLOB(label,0,0,0,0,expectedResult_);

    // propagate inputValue through layers
    solver_->net()->Forward();

    // create BLOB for outputLayer
    Blob<double>* outputLayer = solver_->net()->output_blobs()[0];

    // return the only value in output Layer
    cout << "solver output of forward " << getDataOfBLOB(outputLayer,0,0,0,0) << endl;

    // ------------

    cout << "iter_size" << solver_->param().iter_size() << endl;
    //cout << solver_->param().



    //shared_ptr<caffe::Solver<float> >
    //    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    solver_->Solve();

    // create BLOB for outputLayer
//    Blob<double>* outputLayer = net->output_blobs()[0];

    // return the only value in output Layer
    return getDataOfBLOB(outputLayer,0,0,0,0);
}

// -- vector

/**
 * @brief ANN::train trains the network with the given inputs and expected outputs
 * @param inputValues_ vector of inputValues
 * @param expectedOutputValues_ vector of output values
 * @param solverFile_ solver prototxt file, which defines all parameters and the structure of the net
 *
 * The train function executes a learning to the net by doing the following steps :
 *   1. create a solver object which encapsulate and controls the net defined in solverFile_
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
 */
bool ANN::train (vector<double> inputValues_, vector<double> expectedOutputValues_, const string& solverFile_) {
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
        iFile.open(solverFile_);
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
            return true;
        }
    }
}



/* --- miscellaneous --- */

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
