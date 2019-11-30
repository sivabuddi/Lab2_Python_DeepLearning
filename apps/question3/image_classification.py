import numpy as np
import os
import cv2
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models, layers

labels = os.listdir('natural_images/')
print(labels)

x_data = []
y_data = []

for label in labels:
    pics = os.listdir('natural_images/{}/'.format(label))
    for pic in pics:
        image = cv2.imread('natural_images/{}/{}'.format(label, pic))
        image_resized = cv2.resize(image, (32, 32))
        x_data.append(np.array(image_resized))
        y_data.append(label)

x_data = np.array(x_data)
y_data = np.array(y_data)

x_data = x_data.astype('float32') / 255

print(x_data)

enc = LabelEncoder().fit(y_data)
y_encoded = enc.transform(y_data)

y_categorical = to_categorical(y_encoded)

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_categorical, test_size=0.33)

sequential_model = models.Sequential()
sequential_model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
sequential_model.add(layers.MaxPool2D(pool_size=(2, 2)))
sequential_model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
sequential_model.add(layers.MaxPool2D(pool_size=(2, 2)))
sequential_model.add(layers.Dropout(rate=0.25))
sequential_model.add(layers.Flatten())
sequential_model.add(layers.Dense(256, activation='relu'))
sequential_model.add(layers.Dropout(rate=0.5))
sequential_model.add(layers.Dense(8, activation='softmax'))

sequential_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

history = sequential_model.fit(X_train, Y_train, epochs=15, validation_split=0.2)

# predicting the accuracy of the model
score = sequential_model.evaluate(X_test, Y_test, verbose=1)
print('Loss: %.2f, Accuracy: %.2f' % (score[0], score[1]))
predicted = sequential_model.predict_classes(X_test)
predicted_classes = enc.inverse_transform(predicted)
tested_classes = enc.inverse_transform(np.argmax(Y_test, axis=1, out=None))

# plotting the loss
plot.plot(history.history['loss'])
plot.title('model loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='upper left')
plot.show()
