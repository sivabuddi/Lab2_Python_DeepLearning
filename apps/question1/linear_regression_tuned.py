import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split

data = pd.read_csv('Admission_Predict.csv', header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(data[1:, 0:8], data[1:, 8],
                                                    test_size=0.25, random_state=87)


def create_model():
    sequential_model = Sequential()
    sequential_model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    sequential_model.add(Dense(13, init='normal', activation='softmax'))
    sequential_model.add(Dense(1, activation="linear"))
    sequential_model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy', 'mse'])
    return sequential_model


tensorboard = TensorBoard(log_dir="/tmp/tensor-board/linear-regression", histogram_freq=0, write_graph=True,
                          write_images=True)

model = create_model()
history = model.fit(X_train, Y_train, epochs=30, batch_size=90, callbacks=[tensorboard])
print(model.evaluate(X_test, Y_test, verbose=0))

plot.plot(history.history['loss'])
plot.title('model loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='upper left')
plot.show()
