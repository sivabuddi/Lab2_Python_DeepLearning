import re

import matplotlib.pyplot as plot
import pandas as pd
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import Embedding, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM, SpatialDropout1D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

test_movie_df = pd.read_csv('movie_reviews/test.tsv', delimiter='\t', encoding='utf-8')
train_movie_df = pd.read_csv('movie_reviews/train.tsv', delimiter='\t', encoding='utf-8')

train_movie_df = train_movie_df.drop(columns=['PhraseId', 'SentenceId'])
test_movie_df = test_movie_df.drop(columns=['PhraseId', 'SentenceId'])

train_movie_df['Phrase'] = train_movie_df['Phrase'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x.lower()))
test_movie_df['Phrase'] = test_movie_df['Phrase'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x.lower()))

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_movie_df['Phrase'].values)
X_train = tokenizer.texts_to_sequences(train_movie_df['Phrase'].values)
X_train = pad_sequences(X_train)

tokenizer.fit_on_texts(test_movie_df['Phrase'].values)
X_test = tokenizer.texts_to_sequences(test_movie_df['Phrase'].values)
X_test = pad_sequences(X_train)
print("handing data")

embed_dim = 128
lstm_out = 196


def create_model():
    sequential_model = Sequential()
    sequential_model.add(Embedding(max_fatures, embed_dim, input_length=X_train.shape[1]))
    sequential_model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    sequential_model.add(Dropout(0.5))
    sequential_model.add(Dense(5, activation='softmax'))
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.01 / 15, nesterov=False)
    sequential_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return sequential_model


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train_movie_df['Sentiment'])
Y_train = to_categorical(integer_encoded)
X_TR, X_TST, Y_TR, Y_TST = train_test_split(X_train, Y_train, test_size=0.25, random_state=30)

model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 500

history = model.fit(X_TR, Y_TR, epochs=2, batch_size=batch_size)
score, acc = model.evaluate(X_TST, Y_TST, verbose=2, batch_size=batch_size)

# plotting the loss
plot.plot(history.history['loss'])
plot.title('model loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='upper left')
plot.show()
