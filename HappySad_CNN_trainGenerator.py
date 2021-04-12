# Assignment 4:Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
# Create a convolutional neural network that trains to 100% accuracy on these images, which cancels training upon hitting
# training accuracy of >.999
# Hint -- it will work best with 3 convolutional layers.


import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
!cat /proc/meminfo
# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

# Just visualizing images of the dataset
pp=os.listdir('/tmp/h-or-s')
print(type(pp))
print(pp)
ss=pp[1]
print(ss)
sad_path=os.path.join('/tmp/h-or-s',ss)
print(sad_path)
file_name=os.listdir(sad_path)[10]
print(os.listdir(sad_path)[0])
img_path=os.path.join(sad_path,file_name)
print(img_path)

%matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img = mpimg.imread(img_path)
plt.imshow(img)


# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
          def on_epoch_end(self,epoch,logs={}):
            print("\n debug: ",logs['acc'])    
            if(logs.get('acc')>DESIRED_ACCURACY):
               print("Reached 99% accuracy so cancelling training!")
               self.model.stop_training=True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128,activation='relu'),
                                        tf.keras.layers.Dense(1,activation='sigmoid')
        # Your Code Here
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)# Your Code Here

    # Please use a target_size of 150 X 150.
	# The directory name should be the main directory which contains individual folders of each class, named as per the classes
	# target_size should not contain channel information. Only idth x height is needed.
    train_generator = train_datagen.flow_from_directory('/tmp/h-or-s', target_size=(150,150),batch_size=4,class_mode='binary')
        # Your Code Here)
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(train_generator,steps_per_epoch=20,epochs=15,verbose=1,callbacks=[callbacks])
          # Your Code Here)
    # model fitting
    return history.history['acc'][-1]
	
	# The Expected output: "Reached 99.9% accuracy so cancelling training!""
train_happy_sad_model()