import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenizing words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        #add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
#sort classes
classes = sorted(list(set(classes)))
#documents = combination between patterns and intents
print(len(documents), "documents")
#classes = intents
print(len(classes), "classes", classes)
#words = all words, vocab
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#training data
training = []
#empty array for output
output_empty = [0] * len(classes)
#training set, bag of words for each sentence
for doc in documents:
    bag = [0] * len(words)  # Initialize bag with zeros
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    for i, w in enumerate(words):
        if w in pattern_words:
            bag[i] = 1

    #output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)
#create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:,1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")