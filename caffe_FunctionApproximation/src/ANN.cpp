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
ANN::ANN(const string& modelFile_, const string& trainedFile_) {
    // set processing source
    #ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
    #else
      Caffe::set_mode(Caffe::GPU);
    #endif

    // load network-structure from prototxt-file
    net = new Net<double>(modelFile_,caffe::TEST);

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
