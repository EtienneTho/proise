# Fit and save a basic CNN on MNIST
#
# Written by Etienne Thoret (2020)


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

categories = range(10)
print(x_train.shape)
print(x_test.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
# # Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255.
x_test /= 255.
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

## model to fit
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
print(x_train[1,:,:,:])
model.fit(x=x_train,y=y_train, epochs=3, validation_data=(x_test, y_test), verbose=1)
model.evaluate(x_test, y_test)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



