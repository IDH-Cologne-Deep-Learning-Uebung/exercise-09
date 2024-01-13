import pandas as pd
import numpy as np

# read in CSV file
data = pd.read_csv("exercise-09/data/gmb.csv",encoding = 'latin1')

# the first column of the file contains the sentence number
# -- but only for the first token of each sentence.
# The following line fills the rows downwards.
data = data.fillna(method = 'ffill')

# create a list of unique words and assign an integer number to it
unique_words, coded_words = np.unique(data["Word"], return_inverse=True)
data["Word_idx"] = coded_words
EMPTY_WORD_IDX = len(unique_words)
np.array(unique_words.tolist().append("_____"))
num_words = len(unique_words)+1

# create a list of unique tags and assign an integer number to it
unique_tags, coded_tags = np.unique(data["Tag"], return_inverse=True)
data["Tag_idx"]  = coded_tags
NO_TAG_IDX = unique_tags.tolist().index("O")
num_words_tag = len(unique_tags)

# for verification and inspection, we can inspect the table so far
# data[1:20]

# We are interested in sentence-wise processing.
# Therefore, we use a function that gives us individual sentences.
def get_sentences(data):
  n_sent=1
  agg_func = lambda s:[(w,p,t)
    for w,p,t in zip(
      s["Word_idx"].values.tolist(),
      s["POS"].values.tolist(),
      s["Tag_idx"].values.tolist())]
  grouped = data.groupby("Sentence #").apply(agg_func)
  return [s for s in grouped]

sentences = get_sentences(data)

from keras.utils import pad_sequences
from keras.utils import to_categorical

# find the maximum length for the sentences
max_len = max([len(s) for s in sentences])



#error fix: added dtype

# extract the word index
x = np.array([ np.array([ w[0] for w in s ]) for s in sentences ]     , dtype="object")
# extract the tag index
y = np.array([ np.array([ w[2] for w in s ]) for s in sentences ]     , dtype="object")

# shorter sentences are now padded to same length, using (index of) padding symbol
x = pad_sequences(maxlen = max_len, sequences = x,
  padding = 'post', value = EMPTY_WORD_IDX)

# we do the same for the y data
y = pad_sequences(maxlen = max_len, sequences = y,
  padding = 'post', value = NO_TAG_IDX)

# but we also convert the indices to one-hot-encoding
y = np.array([to_categorical(i, num_classes = num_words_tag) for i in  y])

# split the data into training and test data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=1)


import keras
import tensorflow as tf
from keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Input(shape = (max_len,)))
model.add(layers.Embedding(input_dim = num_words, output_dim = 50, input_length = max_len))
model.add(layers.LSTM(units = 5, return_sequences = True))
model.add(layers.Dense(num_words_tag, activation = 'softmax'))
model.summary()

# We use a different optimizer this time
model.compile(optimizer='Adam',
  loss = 'categorical_crossentropy', metrics = ['accuracy'])


#1d=same shape, 2d= with shape (samples, sequence_length)
#print("y Shape", y_train.shape)  #=(43163, 104, 17)
#sample_weight1 = np.zeros(43163)   #1d
#print(unique_tags)
#1d=; 2d=; 3d=  -> pos

print(y_train[0,])  #
#print(y_train[0,10])
#print(y_train[0,10,10])
print((len(y_train),))
#print(x_test[0,100,]) #


#give specific tags more weight...
#sample_weight1 = np.ones(shape=(len(y_train),))
sample_weight1 = np.full(shape=(len(y_train),), fill_value=0)
sample_weight1[(len(y_train),) == 0] = 0.00000000000001
#sample_weight1[(len(y_train),) == "I-tim"] = 1000.0
#sample_weight1[(len(y_train),) == 1] = 1.5
sample_weight1[(len(y_train),) == 1] = 100.0
#class weight?

history = model.fit(
    x_train, np.array(y_train),
    batch_size = 64,
    epochs = 1,
    verbose = 1,
    sample_weight = sample_weight1
)

model.evaluate(x_test, np.array(y_test))

from sklearn.metrics import classification_report

Y_test = np.argmax(y_test, axis=2)

y_pred = np.argmax(model.predict(x_test), axis=2)


print(classification_report(Y_test.flatten(), y_pred.flatten(), zero_division=0, target_names=unique_tags))