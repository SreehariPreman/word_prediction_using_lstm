import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer from training
with open('rnn-lstm/tokenizer.pickle', 'rb') as handle:
    mytokenizer = pickle.load(handle)

# Load the trained model
model = tf.keras.models.load_model('rnn-lstm/word_generation_model.h5')

input_text = "prime"
predict_next_words = 5

for _ in range(predict_next_words):
    token_list = mytokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = mytokenizer.index_word[predicted[0]]  # Convert index to word using tokenizer
    input_text += " " + output_word

print(input_text)
