#Assignment 2: There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

#Task:Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs
# -- i.e. you should stop training once you reach that level of accuracy.

#Some notes:

#It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
#When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
#If you add any additional variables, make sure you use the same names as the ones used in the class

import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"


# GRADED FUNCTION: train_mnist
def train_mnist():
    
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class mycallback(tf.keras.callbacks.Callback):
          def on_epoch_end(self,epoch,logs={}):
            print("\n debug: ",logs['acc'])    
            if(logs.get('acc')>0.99):           
               print("Reached 99% accuracy so cancelling training!")
               self.model.stop_training=True
           
    # 'logs' is a dictionary. Check out its keys to observe the values of other attributes.
    # In some Tensorflow version, 'acc' might be replaced with 'accuracy'. Hence printing the keys is a safer option
    # to avoid unnecesary errors
    # check out other functions in class 'callback' to add more callbacks as per the requirement.	
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    print(type(x_train))
    print(x_train.shape)
    print(y_train.shape)
    # YOUR CODE SHOULD START HERE
    x_train=x_train[0:10000,:,:]
    print(x_train.shape)
    y_train=y_train[0:10000]
    
    callbacks=mycallback()
    x_train=x_train/255.0
    x_test=x_test/255.0  
    
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                        tf.keras.layers.Dense(512,activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
            ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(# YOUR CODE SHOULD START HERE
        x_train,y_train,epochs=10,callbacks=[callbacks]
              # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
	
train_mnist()	