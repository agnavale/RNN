import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from models import Sequential
from layers import Dense, RNN
from losses import Cross_entropy
from optimizers import Adam

text = [
    "I am not interested in cars or electric appliances .",
]
corpus = [line.split(" ") for line in text]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
# print(tokenizer.word_index)

total_words = len(tokenizer.word_index)+1
print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(corpus)
# padded = pad_sequences(sequences,padding="post")

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

# print(xs.shape)
# print("---------------------------------\n",ys.shape)

# #train Neural Network
model = Sequential()
model.add(RNN(total_words,5))
model.add(Dense(5, total_words,activation = "Softmax"))

adam = Adam(learning_rate=1e-4)
model.compile(loss = 'Cross_entropy', optimizer=adam, metric=None)
model.fit(xs, ys, batch_size = np.size(xs), epochs=30000)


#driver code
while True:
    seed_text = input("Enter sentence:")
    if seed_text == "q00":
        break

    next_words = 10

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_sequence_len, padding='pre')
        token_list = to_categorical(token_list,num_classes= total_words)
        
        predictions = model.predict(token_list)
        predicted = np.argmax(predictions, axis=1)[0]
        
        print(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        seed_text+= " " + output_word
        if output_word == '.':
            break

    print(seed_text)

# print("xs", xs.shape)
# print("ys", ys.shape) 

# rnn = RNN(total_words,15)
# dense = Dense(15, total_words, activation="Softmax")
# opt = SGD(learning_rate=0.00001)
    
# for e in range(100):
#     print(f"{e+1}--------------------------------------------")
#     state = rnn.forward(xs)
#     print("state",state.shape)

#     out = dense.forward(state)
#     print("out",out.shape)

#     L = Cross_entropy()
#     loss = L.forward(ys, out)
#     loss_grad = L.backward(ys, out)
#     print("loss_grad",loss_grad.shape)

#     dl_dht = dense.backward(loss_grad)
#     print("dl_dht",dl_dht.shape)

#     dx = rnn.backward(dl_dht)
#     print("dx", dx.shape)

#     opt.update_parms(rnn)
#     opt.update_parms(dense)

# print("---------------------PREDICTION--------------------------------")
# x = np.reshape(xs[5],(1,max_sequence_len,total_words))
# state = rnn.forward(x)
# out = dense.forward(state)
# print("output", out)
# prediction = np.argmax(out, axis=1, keepdims=False)
# print("prediction",prediction)



# total words:12
# xs (9, 9, 12)
# ys (9, 12)
# state (9, 15)
# out (9, 12)
# loss_grad (9, 12)
# dl_dht (9, 15)
# dx (9, 9, 12)

