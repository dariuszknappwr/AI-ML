# Project summary

[1.Entry](#1entry)

[2.Design of the first version of the application](#2design-of-the-first-version-of-the-application)

[2.1.Dataset](#21dataset)

[2.2.Artificial intelligence model](#22artificial-intelligence-model)

[2.3.Model training results](#23model-training-results)

[2.4.Real time gesture recognition](#24real-time-gesture-recognition)

[2.5.In app gesture recognition results](#25in-app-gesture-recognition-results)

[2.5.1.Results on the existing dataset](#251result-on-the-existing-data-set)

[2.5.2.Results on the new dataset](#252results-on-the-new-dataset)

[3.Design of the second version of the application with the ResNet50 model](#3design-of-the-second-version-of-the-application-with-the-resnet50-model)

[3.1.A new model of artificial intelligence](#31a-new-model-of-artificial-intelligence)

[3.2.Dataset](#32dataset)

[3.3.Gesture detection application](#33gesture-detection-application)

[3.4.Application performance results](#34application-performance-results)

[4.New application design with the ms coco model](#4new-application-design-with-the-ms-coco-model)

[4.1.MS COCO artificial intelligence model](#41ms-coco-artificial-intelligence-model)

[4.2.Data preparation](#42data-preparation)

[4.3.Model training results](#43model-training-results)

[4.4.Launching the application](#44launching-the-application)

[5.Conclusions](#5conclusions)

[References](#references)

#


# 1.Entry

The aim of the work was to design and implement an application that recognizes gestures. The designed application has a trained deep network model that is able to recognize 6 gestures: _zero_, _one_, _two_, _three_, _four_, _five_. The _zero_ gesture is performed by joining the thumb and index finger, while the remaining gestures are performed by straightening the appropriate number of fingers.

The application requires one of the popular operating systems, an installed Python interpreter and a camera to work. The application is displayed in the operating system window. The project used a pre-trained image recognition model, which was trained with our own photos.

# 2.Design of the first version of the application

## 2.1.Dataset

The first version of the application aimed to detect gestures from pre-processed images. The image from the camera was subjected to edge detection operations, and then the artificial intelligence model was trained and tested on a given set of photos. Example photos are shown in Figure 1.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/0d457a3e-2fd6-4fee-b14a-2428a990e274)


_Figure 1 Some photos of the training set._

## 2.2.Artificial intelligence model

In order to detect the context from the photo, an artificial intelligence model was built, presented in Listing 1. In the second line of the code, the model accepted photos with a resolution of 256 by 256 pixels. In the third line of code, the model reduced the photo twice in each dimension to omit unnecessary information and increase the speed of photo processing.

There are two convolutional layers in lines 4-5. These layers have a high probability of learning to recognize edges from single pixels, and then should learn to recognize complex shapes from edges. Then the image is repeatedly reduced and subsequent layers are added to learn to recognize the shapes of the image.

Line 13 uses the Dropout layer, which is active only during the training process and randomly resets some neurons. In this case, this layer turns off 25% of the neurons. Adding this layer is intended to make it more difficult for the model to learn, which means that the model will not perfectly adapt to the training data and will be able to work with more diverse data in the future.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/309d3ace-2a19-4719-9d4b-7229e5c56db8)

_Listing 1 An artificial intelligence model that recognizes context in photos._

## 2.3.Model training results

_After training, the model was tested on new images it had never seen before. The model detected gestures with an efficiency of 99.06%._

Figure 2 plots the performance of the model after each training iteration.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/472a25e2-cbab-4d99-8cb0-c14614937284)


_Figure 2 Artificial intelligence model training performance charts._

Listing 2 shows the code that, using the matplotlib library, contains information on the graphs about the number of training epochs, the value of the loss function and the effectiveness of training.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/72479f4a-8e27-4dec-bd33-a77d7151ff79)


_Listing 2 Creating graphs of the loss function and training effectiveness._

Figure 3 shows some misrecognized gestures after the training process. The label placed above the photo describes the model's prediction result and how many fingers actually appear in the photo. In all the photos, only one finger was incorrectly recognized, most often it was the thumb.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/a711d820-a308-4050-b232-32545caabc72)

_Figure 3 Selected incorrectly recognized gestures after the training process._

Figure 4 shows the confusion matrix of the trained AI model after verification on test images. The x-axis contains numbers symbolizing the true number of fingers shown in the photo, and the y-axis shows the model prediction results. All values on the x = y diagonal mean that the model made a correct prediction. Listing 3 shows the code that generated the chart.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/ae07ae9b-9d82-4c2b-8b02-081b64281285)

_Figure 4 Confusion matrix of the trained model_

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/416315dc-dab6-4d43-92e8-f08ff5a7c692)

_Listing 3 Code creating the confusion matrix of the trained model._

## 2.4.Real-time gesture recognition

The application was intended to work with the laptop's built-in camera. To use the trained model, the camera photos had to be converted into data that was as similar as possible to the data used in the model training process. An application was created that used a laptop camera, recorded a single frame from the camera, processed it, made predictions and displayed the prediction result. Figure 5 shows the application view.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/d57298d3-9989-4767-9a24-5cd5fa38acaa)

_Figure 5 Application window that recognizes gestures in real time._

Listing 4 shows the application code. The application runs in an infinite loop until the user presses the _q_ key. At the beginning of the loop, the application reads a single frame from the camera. On line 6 the image is converted to grayscale. Line 7 is given a slight blur to remove minor noise. In line 8, an edge detection filter is applied. Then the frame size is adjusted to the input accepted by the model and the prediction process takes place. The prediction result is visible in the application window.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/9e553cf4-28f9-4fe2-a287-39592c25da7a)

_Listing 4 The code responsible for launching the application._

## 2.5.In-app gesture recognition results

### 2.5.1.Result on the existing data set

Figure 6 shows the effect of the filters. With appropriate lighting and background, the photos are almost identical to the photos from the training set. The application was tested and worked correctly only when the camera image was very close to the training set. There were no items in the photos in the training set that would interfere with the learning process. The application worked correctly only in the case of gestures read against a uniform white background, without the presence of other objects in the frame.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/423d5c09-ab2d-4941-b5ca-1738cae7846f) ![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/f79952ae-7720-42be-a536-5ae7cd85ce20)

_Figure 6 Sample photos from the camera after passing through filters._

### 2.5.2.Results on the new dataset

Attempts were made to retrain the model. For this purpose, an application was created, presented in Listing 4. It downloaded a single frame from the camera and saved it to a directory on the disk. After each series, the photos were manually filtered and some of them were removed. As Figure 7 shows, several thousand photos were obtained this way.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/dac0a2f2-4255-454a-ac79-10269d5fcd31)

_Figure 7 The number of photos created to train the model_

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/bac62d3a-bf11-4d86-a2bc-69351907764d)

_Listing 4 A script that saves a series of photos from the camera to a directory_

Unfortunately, it was not possible to obtain the appropriate accuracy of the model. We tried to train the model in several ways:

- Using only the new training unit
- Using both a new and an old training unit
- By modifying the blur and edge search parameters.
- By modifying the model architecture, adding more Dropout layers

None of the methods brought the expected results. As Figure 8 shows, the model accuracy was only 19% despite the long training time.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/e2030aac-e632-4de9-bd7e-0994d663d7a5)

_Figure 8 The result of training the model on a new data set._

# 3.Design of the second version of the application with the ResNet50 model

## 3.1.A new model of artificial intelligence

It was decided to study the literature on image recognition using deep networks. The ResNet50 model was found (He, Zhang, Ren, & Sun, 2016), whose name comes from the use of 50 layers of convolutional networks. It was decided to choose this model due to achieving very high image recognition efficiency thanks to the complex architecture of the model.

ResNet50 is a very popular model, so it is already pre-implemented in many libraries. The Keras library allows the use of multiple models together with pre-learned model weights [4]. Training a model involves training it with your own data set. Figure 9 shows examples of available models.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/aec96a60-fd07-4ee6-9f05-c842941bf8dc)

_Figure 9 Sample models available in the Keras library_

## 3.2.Dataset

The ResNet50 model required color photos, it did not want to accept black and white photos as input, and in addition, the photos had to have a resolution of exactly 224x224 pixels.

Unfortunately, it was not possible to find any comprehensive dataset that would fit the topic of gesture recognition. While the resolution of the photos could be rescaled, all photos used for training the first version of the model were saved in black and white.

In order to test the new model, a new data set was created consisting of several hundred color photos that were saved in the source resolution. Figure 10 shows sample photos used to train the model.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/c882aa40-1f3e-4462-ac4a-a74d7072262c) ![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/ee761b8e-5dcf-493c-a2ef-38c02a5d47a4)

_Figure 10 Example photos taken for the ResNet50 model._

Listing 5 shows a script that loads photos from a given folder and adapts the photos to the requirements of the ResNet50 model. Photos are also labeled based on the name of the directories in which they are located. All photos with the _jeden_ gesture are placed in a directory called _ONE_, with the _two_ gesture in the _TWO_ directory, etc.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/806dfd14-9620-43f5-8f44-9d3588a7de27)

_Listing 5 Preparing photos from disk to be loaded into the ResNet50 model_

## 3.3.Gesture detection application

Listing 6 shows the loading of the Resnet50 model from the Keras library. The model is loaded without the last output layer, and then its own output layer is created and added to the model. This is to adapt the model to the appropriate number of predictions that the program assumes. Listing 7 shows the updated application adapted to work with the new data model.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/d3a0bb27-68dc-41ba-a20c-ca408824b016)

_Listing 6 Loading the ResNet50 model from the Keras library._

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/f221e30f-d22a-462b-b620-7b0ca5406e71)

_Listing 7 Application using the ResNet50 model_

## 3.4.Application performance results

The results of the application were very similar to the previously obtained results, presented in Figure 8. The use of the new model did not contribute to improving the effectiveness of gesture recognition. This was probably due to the poor data set. In this situation, it was decided to review the literature on gesture recognition again.

# 4.New application design with the MS COCO model

## 4.1.MS COCO artificial intelligence model

A literature review found the Microsoft COCO: Common Objects in Context model (Lin et al., 2014). This model makes a different assumption than previous models. Instead of analyzing the entire photo and trying to learn edge shapes, this model tries to break the photo into pieces and extract context from it. The model was initially trained on 238 thousand. Photos with a total of 2.5 million labels. Figure 11, marking (d) shows the principle of operation of the model.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/1eeb840b-5240-4c9d-9a10-1eb7f0cd312e)

_Figure 11 a) image classification, b) finding an object and surrounding it with a rectangle, c) semantic separation at the level of individual pixels, d) separation of all object instances._

Figure 12 shows the Microsoft site where pre-trained models can be found. These models are large and require a significant amount of time to train, so we set out to find research on the differences between each model.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/fb3300a9-0f75-4e5f-b60e-523c208d783c)

_Figure 12 Pre-trained models available on Tensorflow._

Figure 13 shows a comparison made by one of the users [3]. The user tested the detection of US coins on several hundred photos. The comparison results are shown in Figure 13, which shows the number of frames per second (FPS) obtained on the Raspberry Pi camera and the coin detection efficiency. The effectiveness was calculated based on statistics from several hundred photos. It was decided to choose the SSD-MobileNet-v2-FPNLite model due to its acceptable performance and the best effectiveness.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/743e2547-9fe9-4b1f-b4fb-2750c029f873)

_Figure 13 Model comparison from Tensorflow._

## 4.2.Data preparation

Once the model was selected, the data needed to be prepared. All color photos were marked using the LabelImg program, which allowed you to select a part of the photo and apply labels to it. Figure 14 shows the effect of the program.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/44b1595d-97f1-4a98-a097-624329c66a1a)

_Figure 14 Applying a label in LabelImg._

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/1b2dd1bd-c722-4425-8fa4-9c5bceaf695c)

_Figure 15 Label from LabelImg as an xml file._

You can select multiple labels in one photo that will participate in the model training process. Figure 15 shows the saved label in the form of an xml file. A total of 428 photos were tagged.

## 4.3.Model training results

After preparing the data, the training process began. It was decided to train the model until the value of the loss function dropped to an appropriate level. Figure 16 shows the loss function curve during 3.5 hours of training the model, while Figure 17 shows the learning process curve for this model.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/0c7e302d-e57e-4bbb-a708-d06c77b9826d)

_Figure 16 Loss function curve._

  ![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/5bc506dd-434b-435f-80d1-d5b8ddc76ae7)


_Figure 17 Learning curve._

The model results were satisfactory, both the gesture and the position of the rectangle were predicted with high efficiency, as shown in Figure 18 and Figure 19.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/bedee42a-ae44-4d2a-a764-3983d193875d)


_Figure 18 Result of the trained model for gesture zero._

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/d399baad-cef2-4e77-acbd-bcadc3f3ad8f)


_Figure 19 Result of the trained model for gesture three._

Listing 8 shows a script using the trained model to detect gestures. Line 124 captures the time that is used to calculate frames per second. In lines 126-134, a single frame from the camera is captured and its parameters are processed to those expected by the model.

In lines 136-137, prediction takes place and the model detects gestures, and then in lines 139-141 the following are written: coordinates of the detected gesture, class of the detected gesture, model confidence in the detected gesture. The model usually returns many detected objects, so they are processed in a loop and those with too low confidence are skipped.

In lines 146-149, coordinates are taken, which in line 151 are used to draw a rectangle on the screen. The next operations ensure that the rectangle is drawn inside the application window and the number of frames per second is calculated.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/94bba7e3-2438-4122-b5ac-ea45358354b6)

_Listing 8 A script using the trained model to detect gestures._

In the case of object detection models, a popular measure of model quality is Mean Average Precision (mAP). Calculating the average precision involves calculating the precision for each class (each gesture) for the threshold value: 0.5; 0.55; 0.6; 0.65; 0.7; 0.75; 0.8; 0.85; 0.9; 0;95 and then calculating the average of all these values. The threshold value indicates the certainty of the model, e.g. a threshold of 0.5 means that the model is 50% sure that a given gesture is present in a given location.

You can find ready-made scripts on the Internet that calculate the average precision of the model. Figure 20 shows the calculated average precision of the model. The average precision should be above the 50% level, and an average precision above the 90% level indicates an excellent result. Obtaining a value of 78.91% means that gesture detection works at a good level.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/50095fe0-e904-4e9a-a3df-e35e1ddfe635)

_Figure 20 The mean Average Precision (mAP) score for the model._

The recognition of the _two_ gesture was definitely the worst, because the model often confused this gesture with the _one_ or _three_ gesture, as shown in Figure 21. Only wide spread of the fingers in the _two_ gesture resulted in the correct detection of the gesture.

![obraz](https://github.com/dariuszknappwr/AI-ML/assets/127883702/abe4f2c3-45e2-4f42-b2b6-c389171400d6)

_Figure 21 Incorrect gesture detection by the model._

## 4.4.Launching the application

The application can be launched after running the command in the console

- python3 .\TFLite\_detection\_webcam.py --modeldir=custom\_model\_lite

, where the --modeldir parameter is the path to the folder with the trained model. Python requires several libraries to be installed. In Linux and Windows environments, this is possible using the commands:

- python3 -m pip install numpy
- python3 -m pip install opencv-python
- python3 -m pip install tensorflow

# 5.Conclusions

The first version of the application focused on gesture recognition using pre-processed camera images. The model's performance results were promising, but there were limitations related to the uniform background and lighting. On the new dataset, especially with a more diverse background, the performance dropped significantly. It was decided to use the new ResNet50 model. This particular model was chosen due to its high efficiency in image recognition. The problem was finding a dataset, so a new dataset was created. Despite the model change, the application's effectiveness has not improved significantly. Then it was decided to use the Microsoft COCO model, which focuses on the context of objects. This model achieved an efficiency of 78.91%, which is a satisfactory result. Further work may include acquiring more training data with a variety of backgrounds and lighting.

# References

[1] He, K., Zhang, X., Ren, S. i Sun, J. (2016). _Deep Residual Learning for Image Recognition_. https://ieeexplore.ieee.org/document/7780459

[2] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., . . . Zitnick, C. L. (2014). _Microsoft COCO: Common Objects in Context_. https://link.springer.com/chapter/10.1007/978-3-319-10602-1\_48

[3] [https://www.ejtech.io/learn/tflite-object-detection-model-comparison](https://www.ejtech.io/learn/tflite-object-detection-model-comparison), comparasion of MS COCO models.

[4] [https://datagen.tech/guides/computer-vision/resnet-50/](https://datagen.tech/guides/computer-vision/resnet-50/), ResNet50 model introduction.
