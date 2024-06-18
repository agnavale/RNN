import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from models import Sequential
from layers import Dense, RNN
from optimizers import Adam

text = [
    "I am not interested in cars or electric appliances .",
    "And my ankle and knee are in pain It is very painful .",
    "I understand it is a dream ."
]
corpus = [line.split(" ") for line in text]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
# print(tokenizer.word_index)

total_words = len(tokenizer.word_index)+1
# print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(corpus)
input_sequences = []
labels = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i]
        input_sequences.append(n_gram_sequence)
        labels.append(token_list[i])

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,maxlen = max_sequence_len, padding = 'pre'))

xs = to_categorical(input_sequences, num_classes=total_words)
ys = to_categorical(labels, num_classes=total_words)

# #train Neural Network
model = Sequential()
model.add(RNN(total_words,5))
model.add(Dense(5, total_words,activation = "Softmax"))

adam = Adam(learning_rate=1e-5)
model.compile(loss = 'Cross_entropy', optimizer=adam, metric=None)
model.fit(xs, ys, batch_size = np.size(xs), epochs=1000)


#driver code
seed_text = "I am not interested in"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen = max_sequence_len, padding='pre')
    token_list = to_categorical(token_list,num_classes= total_words)
    
    predictions = model.predict(token_list)
    predicted = np.argmax(predictions, axis=1)[0]
    
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    
    seed_text+= " " + output_word
    if output_word == '.':
        break
        
print(seed_text)
