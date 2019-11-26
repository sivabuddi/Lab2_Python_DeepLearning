import keras
import pandas as pd
import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

heart_df = pd.read_csv("heart.csv")

# scaling not required for
# FPS (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# Sex (1 = male; 0 = female)
# RESTECH (resting electrocardiographic results)
# EXANG (exercise induced angina (1 = yes; 0 = no))

# scaling the features.
heart_df.age = (heart_df.age - np.min(heart_df.age)) / (np.max(heart_df.age) - np.min(heart_df.age))
heart_df.cp = (heart_df.cp - np.min(heart_df.cp)) / (np.max(heart_df.cp) - np.min(heart_df.cp))
heart_df.trestbps = (heart_df.trestbps - np.min(heart_df.trestbps)) / (
        np.max(heart_df.trestbps) - np.min(heart_df.trestbps))
heart_df.chol = (heart_df.chol - np.min(heart_df.chol)) / (np.max(heart_df.chol) - np.min(heart_df.chol))
heart_df.thalach = (heart_df.thalach - np.min(heart_df.thalach)) / (np.max(heart_df.thalach) - np.min(heart_df.thalach))
heart_df.oldpeak = (heart_df.oldpeak - np.min(heart_df.oldpeak)) / (np.max(heart_df.oldpeak) - np.min(heart_df.oldpeak))
heart_df.slope = (heart_df.slope - np.min(heart_df.slope)) / (np.max(heart_df.slope) - np.min(heart_df.slope))
heart_df.ca = (heart_df.ca - np.min(heart_df.ca)) / (np.max(heart_df.ca) - np.min(heart_df.ca))
heart_df.thal = (heart_df.thal - np.min(heart_df.thal)) / (np.max(heart_df.thal) - np.min(heart_df.thal))

values_scaled = np.asarray(heart_df.values[1:, :], dtype=np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(values_scaled[:, 0:13], values_scaled[:, 13], test_size=0.25, random_state=87)


Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

model = Sequential()
model.add(Dense(output_dim=10, input_shape=(13,), init='normal', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# tensorboard graph genertion
tensorboard = TensorBoard(log_dir="/tmp/tensor-board/logistic-regression", histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(X_train, Y_train, nb_epoch=10, batch_size=60, callbacks=[tensorboard])

# predicting the accuracy of the model
score = model.evaluate(X_test, Y_test, verbose=1)
print('Loss: %.2f, Accuracy: %.2f' % (score[0], score[1]))

# plotting the loss
plot.plot(history.history['loss'])
# plt.plot(history.history['test_loss'])
plot.title('model loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='upper left')
plot.show()
