# Understanding neural network with Tensorflow


- house_price_prediction.py- The first assignment is a "Hello World " assignment for Neural Network
wherein, we fit a linear regression equation, with few examples.
Play with the code by adding more values of x and y and observe the change in accuracy.

- MNIST_digit_callback.py- This assignment shows the usage of callbacks in Tensorflow. A 2-layer dense-layer neural network
is used to classify MNIST digits. Check other keys of 'history' of model such as 'loss', to implement callback.

- MNIST_digit_CNN.py- This assignment shows the usage of single Convolution layer and a maxpool layer for classification. A callback
function is also implemented to stop the training. Check out performance, training time, model parameters by varying number of filters in 
convolution layer. Also observe the variation by changing the filter size.

- HappySad_CNN_trainGenerator.py - This assignment is based on a small database of 80 images with happy and sad face. Interestingly,
 one can check that database is actually smileys of happy and sad mode. CNN with multiple convolution layers is used for training.
 One can observe variation in performance by successively increasing the convolution layers as well as number of filters. It also
 demonstrates the usage of inbuilt 'ImageGenerator' in keras which only needs the name of main directory of training data, assuming that the
 subfolders within it are named with respect to the classes.

* Note:This repo contains programming assignments done by me during the Coursera course offered by DeepLearning.AI