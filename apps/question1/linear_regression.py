import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('Admission_Predict.csv', header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(data[1:, 0:8], data[1:, 8],
                                                    test_size=0.25, random_state=87)


def createmodel():
    model = Sequential()
    model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    model.add(Dense(13, init='normal', activation='relu'))
    model.add(Dense(1, activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mse'])
    return model


tensorboard = TensorBoard(log_dir="tensor_board/1", histogram_freq=0, write_graph=True, write_images=True)

model = createmodel()
history = model.fit(X_train, Y_train, epochs=15, batch_size=120, callbacks=[tensorboard])
print(model.evaluate(X_test, Y_test, verbose=0))

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
