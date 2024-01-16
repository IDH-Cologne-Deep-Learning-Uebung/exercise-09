import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers
from sklearn.metrics import classification_report


data = pd.read_csv("data/gmb.csv", encoding='latin1')
data = data.fillna(method='ffill')


unique_words, coded_words = np.unique(data["Word"], return_inverse=True)
data["Word_idx"] = coded_words
EMPTY_WORD_IDX = len(unique_words)
np.array(unique_words.tolist().append("_____"))
num_words = len(unique_words) + 1


unique_tags, coded_tags = np.unique(data["Tag"], return_inverse=True)
data["Tag_idx"] = coded_tags
NO_TAG_IDX = unique_tags.tolist().index("O")
num_words_tag = len(unique_tags)


def get_sentences(data):
    agg_func = lambda s: [(w, p, t)
                         for w, p, t in zip(
            s["Word_idx"].values.tolist(),
            s["POS"].values.tolist(),
            s["Tag_idx"].values.tolist())]
    grouped = data.groupby("Sentence #").apply(agg_func)
    return [s for s in grouped]

sentences = get_sentences(data)


max_len = max([len(s) for s in sentences])


x = np.array([np.array([w[0] for w in s]) for s in sentences])
y = np.array([np.array([w[2] for w in s]) for s in sentences])


x = pad_sequences(maxlen=max_len, sequences=x, padding='post', value=EMPTY_WORD_IDX)


y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=NO_TAG_IDX)
y = np.array([to_categorical(i, num_classes=num_words_tag) for i in y])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)


sample_weights = compute_sample_weight('balanced', y_train.argmax(axis=2).flatten())


model = models.Sequential()
model.add(layers.Input(shape=(max_len,)))
model.add(layers.Embedding(input_dim=num_words, output_dim=50, input_length=max_len))
model.add(layers.LSTM(units=5, return_sequences=True))
model.add(layers.Dense(num_words_tag, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    x_train, np.array(y_train),
    sample_weight=sample_weights,
    batch_size=64,
    epochs=5,
    verbose=1
)


model.evaluate(x_test, np.array(y_test))


Y_test = np.argmax(y_test, axis=2)
y_pred = np.argmax(model.predict(x_test), axis=2)
print(classification_report(Y_test.flatten(), y_pred.flatten(), zero_division=0, target_names=unique_tags))
