OBJECTIVE:
The Convolutional Neural Network is fed and trained with images of animals and the goal here is to correctly predict the name of animal which is given as input by the user for prediction

PREPROCESSING:
To avoid overfitting we apply transformations to the training set like geometrical transformation like changing geometry,some rotations,zoom in,zoom out and shifiting pixels
Its also called Image augmentation-transforming so that train set does not overtrains(overfitting)

LIBRARIES USED:
Tensorflow(keras),Image Data generator for preprocessing,Pandas

STEPS:
1)Build a sequential model with 2 hidden layers
2)Select the appropriate number for input shape and filters
3)Next the model needs to be passed into a pooling layer to select spatially important features and eliminate others.
4)Repeat step 2 and 3 for the second hidden layer
5)Flatten the output from the second hidden layer of CNN to give as input to the first ANN layer.
6)Compile and fit the model with appropriate train and test set.
7)Compare the actual with predicted and calculate the error rate.


EVALUATION METRIC:
Accuracy
