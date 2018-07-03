# Reference: https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac
# skip-gram implementation of word2vec

import numpy as np
import tensorflow as tf

corpus = "He is the king. The king is royal. She is the royal queen"

corpus = corpus.lower()

word2int = {} #mapping of word to integer
int2word = {} #mapping of integer to word
index = 0
window_size = 2

def to_one_hot(index, size):
    array = np.zeros(size)
    array[index] = 1
    return array

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

def skip_gram_model(sentences):
    # we tried to predict a neighbouring words given a word
    data = []
    #creating list of word and true label tuples
    for sentence in sentences:
        #sentences.append(sentence)
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - window_size, 0) : min(word_index + window_size, len(sentence)) + 1] :
                if nb_word != word:
                    data.append([word, nb_word])
    return data

def bag_words_model(sentences):
    # use neighbouring words of a middle word as inputs and asked the network to predict the middle word
    data = []
    #creating list of word and true label tuples
    for sentence in sentences:
        #sentences.append(sentence)
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - window_size, 0) : min(word_index + window_size, len(sentence)) + 1] :
                if nb_word != word:
                    data.append([nb_word, word])
    return data
 

#creating a map for word to int and int to word to be used later.
for word in corpus.split():
    if word!='.' and not word2int.has_key(word):
	word2int[word] = index
	int2word[index] = word 
	index+=1

vocab_size = len(word2int)

sentences = []
data = []

for sentence in corpus.split('.'):
    sentences.append(sentence.split())

print sentences

#data = skip_gram_model(sentences)
data = bag_words_model(sentences)

#convert words and labels to one-hot vector format

x_train = []
y_train = []

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))	

#convert it to num py array
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print x_train
print y_train

#making placeholders for x_train and y_train

x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5 # you can choose your own number

W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias

hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))

prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

#input_one_hot  --->  embedded repr. ---> predicted_neighbour_prob
#predicted_prob will be compared against a one hot vector to correct it.

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!

# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000

# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

print sess.run(W1)
print sess.run(b1)

vectors = sess.run(W1 + b1)
print(vectors)

print(vectors[ word2int['queen'] ])

print(int2word[find_closest(word2int['king'], vectors)])
print(int2word[find_closest(word2int['queen'], vectors)])
print(int2word[find_closest(word2int['royal'], vectors)])
