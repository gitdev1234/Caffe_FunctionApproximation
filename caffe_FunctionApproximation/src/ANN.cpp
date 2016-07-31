#include "ANN.h"

ANN::ANN(const string& modelFile_, const string& trainedFile_) {
    // set processing source
    #ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
    #else
      Caffe::set_mode(Caffe::GPU);
    #endif

    // load network-structure from prototxt-file
    net = new Net<float>(modelFile_,caffe::TEST);

    if (trainedFile_ != "") {
        net->CopyTrainedLayersFrom(trainedFile_);
    }
}

float ANN::forward(float inputValue_) {

    // create BLOB for input layer
    Blob<float>* inputLayer = net->input_blobs()[0];

    // set dimesions of input layer
    // --> for normal caffe works with images, therefore the data
    // --> typically is 4 dimensional
    // --> numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
    // --> in this case we use 1-dimensional data, therefore the data-dimension is 1*1*1*1
    int num      = 1;
    int channels = 2;
    int height   = 1;
    int width    = 1;
    vector<int> dimensionsOfInputData = {num,channels,height,width};
    inputLayer->Reshape(dimensionsOfInputData);

    // forward dimension change to all layers.
    net->Reshape();

    // insert inputValue into inputLayer
    setDataOfBLOB(inputLayer,0,0,0,0,inputValue_);

    // propagate inputValue through layers
    net->Forward();

    // create BLOB for outputLayer
    Blob<float>* outputLayer = net->output_blobs()[0];

    // return only value in output Layer
    return getDataOfBLOB(outputLayer,0,0,0,0);
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
void ANN::setDataOfBLOB(Blob<float> *blobToModify_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_, float value_) {
    // check if index is invalid
    if ( (indexNum_     < 0) || (indexNum_     > blobToModify_->num()      - 1) ||
         (indexChannel_ < 0) || (indexChannel_ > blobToModify_->channels() - 1) ||
         (indexHeight_  < 0) || (indexHeight_  > blobToModify_->height()   - 1) ||
         (indexWidth_   < 0) || (indexWidth_   > blobToModify_->width()    - 1) ){
        cout << "Error : please use valid indexes!" << endl;
    } else {
        // create a pointer, that points to the first value inside the blobToModify
        float* pointerToBlobValue = blobToModify_->mutable_cpu_data();

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
float ANN::getDataOfBLOB(Blob<float> *blobToReadFrom_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_) {
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
