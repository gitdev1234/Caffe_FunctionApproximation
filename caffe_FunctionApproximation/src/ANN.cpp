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

void ANN::forward(float inputValue_) {


    Blob<float>* inputLayer = net->input_blobs()[0];

    // for normal caffe works with images, therefore the data
    // typically is 4 dimensional
    // numberOfImages * numberOfColorChannels * numberOfPixelsInDirectionOfHeight * numberOfPixelsInDirectionOfWidth
    // in this case we use 1-dimensional data, therefore the data-dimension is 1*1*1*1
    int num      = 3;
    int channels = 2;
    int height   = 2;
    int width    = 2;
    vector<int> dimensionsOfInputData = {num,channels,height,width};
    inputLayer->Reshape(dimensionsOfInputData);

    // create a pointer, to data inside input Layer
    float* inputData = inputLayer->mutable_cpu_data();

    int indexNum     = 0;
    int indexChannel = 1;
    int indexHeight  = 1;
    int indexWidth   = 1;
    int addressIncrement = indexNum * inputLayer->channels() * inputLayer->height() * inputLayer->width();
        addressIncrement +=  indexChannel * inputLayer->height() * inputLayer->width();
        addressIncrement +=  indexHeight * inputLayer->width();
        addressIncrement +=  indexWidth;


    setDataOfBLOB(inputLayer,indexNum,indexChannel,indexHeight,indexWidth,55.6);
    cout << "value : " << getDataOfBLOB(inputLayer,indexNum,indexChannel,indexHeight,indexWidth) << endl;
    cout << "count : " << inputLayer->count() << endl;
    cout << "value orig : " << inputLayer->data_at(0,1,1,1) << endl;

    Blob<float> *temp = inputLayer;
    temp->Reshape(3,3,2,2);
    cout << "value : " << getDataOfBLOB(temp,indexNum,indexChannel,indexHeight,indexWidth) << endl;
    cout << "count : " << temp->count() << endl;
    cout << "value orig : " << inputLayer->data_at(0,1,1,1) << endl;


    /* Forward dimension change to all layers. */
    net->Reshape();
    cout << inputLayer->data_at(1,-1,1,1);
    //inputLayer->set_cpu_data(inputValue_);


    net->Forward();
    Blob<float>* outputLayer = net->output_blobs()[0];
    //cout << outputLayer->data_at(1,1,1,1);
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
 * NOTICE : If an invalid index is handed into the function, the function is just doing nothing.
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

float ANN::getDataOfBLOB(Blob<float> *blobToReadFrom_, int indexNum_, int indexChannel_, int indexHeight_, int indexWidth_) {
    // create a pointer, that points to the first value inside the blobToReadFrom
    float* pointerToBlobValue = blobToReadFrom_->mutable_cpu_data();

    int addressIncrement  = indexNum_     * blobToReadFrom_->channels() * blobToReadFrom_->height() * blobToReadFrom_->width();
        addressIncrement += indexChannel_ * blobToReadFrom_->height()   * blobToReadFrom_->width();
        addressIncrement += indexHeight_  * blobToReadFrom_->width();
        addressIncrement += indexWidth_;

    pointerToBlobValue += addressIncrement;

    return *pointerToBlobValue;

}
