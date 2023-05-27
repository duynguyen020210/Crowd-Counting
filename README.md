
# Crowd-counting

This notebook demonstrates several approaches to automatic counting of people using images from indoor video cameras in a shopping center.

## Dataset
The problem of object counting is a difficult problem in which you have an image and have to count all the certain objects on that image, this becomes intractable in the context of big data and internet of things when you have videos tracking and need to count the objects in every frame. This problem can be achieved by an automatic process like a machine learning algorithm that receives a image as input and gives the number of some object of interest in the image (discrete value). For that different approaches can be done, the typical and easier is consider the number of element in the image as a label and transform it to some classification problem, other transform to a regression problem and other ones use a fully convolutional architecture in where the final convolutional output can be consider the numbers of object in that region and then sum up.


## Solution

Out-of-the-box solution: **EfficientDet** object detection model loaded from TensorFlow Hub. The model is capable of detecting a wide range of objects returning predicted class, bounding box coordinates and confidence score for each object. Benefits: doesn't require training, multiple models are available, could be easily deployed on various devices. Drawbacks: model is prone to errors when detecting multiple objects, objects partly accluded or located at the background, model is difficult to retrain and fine-tune.

Transfer learning: **InceptionResNetV2** as feature extractor with a new regression head. Benefits: model is relatively easy to fine-tune for a new task while retaining the useful knowledge of the original classifier. Drawbacks: despite higher accuracy compared to the previous solution, this model is not perfect and could not be used in environments where high precision is important.

## Algorithm
### Part 1: Using out-of-the-box model

EfficientDet model: SSD with EfficientNet + BiFPN feature extractor, shared box predictor and focal loss, trained on COCO 2017 dataset. Several models of various sizes could be found at TF Hub. We will use the smallest model d0.

Model inputs: a three-channel image of variable size - a tf.uint8 tensor with shape [1, height, width, 3] with values in [0, 255].

Algorithm
Extract example images from .jpg files and convert to tf.Tensor without resizing or preprocessing.
Visualize model predictions overlaying predicted bounding boxes over the example images.
Select the minimum confidence score to improve the model accuracy.
Check the accuracy on randomly selected subset of images:
EfficientDet model cannot process batches of images, so we process them one by one using multiprocessing for time optimization.
Postprocess the model output for each image counting the number of objects identified as "person" with a selected confidence threshold.

### Part 2: Transfer learning¶

We will load InceptionResNetV2 model from Keras applications and freeze the original weights. The model will be trained with a new regression head. The learning rate will be adjusted whenever validation loss is getting worse.

The original model was trained on images of size 299 x 299. We will resize the images accordingly using padding to avoid distorting the objects. To compensate for small number of training samples we will apply various image augmentation techniques randomly changing brightness, contract, saturation and hue and flipping the images left-to-right.

## Results
