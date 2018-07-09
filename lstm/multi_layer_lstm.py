import numpy as np
import tensorflow as tf

def bag_of_word(sentences):

    data = []
    for sentence in sentences:
	for index in xrange(0, len(sentence)-1):
	    data.append([sentence[index], sentence[index+1]])
        data.append([sentence[len(sentence)-1], sentence[len(sentence)-1]])

    return data

def to_one_hot(index, size):

    array = np.zeros(size)
    array[index] = 1
    return array


sentence_1 = ['my','cat','sleeps','on','my','bed', 'EOS']
sentence_2 = ['a', 'dog', 'sleeps', 'on', 'the', 'floor', 'EOS']

sentences = []
sentences.append(sentence_1)
sentences.append(sentence_2)

word2int = {}
int2word = {}

index = 0
for sentence in sentences:
    for word in sentence:
	if not word2int.has_key(word):
            word2int[word] = index
	    int2word[index] = word
	    index+=1

print word2int
print int2word

vocab_size = len(word2int)
data = []
data = bag_of_word(sentences)

x_train = []
y_train = []

#prepare data fo training
for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))

#convert it to numpy array
X_train = tf.convert_to_tensor(np.asarray(x_train),dtype=tf.float32)
Y_train = tf.convert_to_tensor(np.asarray(y_train),dtype=tf.float32)

print X_train
print Y_train

print X_train.shape

#training variable setup for lstm
plot_loss=[]
num_hidden = 24
num_steps = 7
dict_length=len(word2int)
batch_size = 2
num_layers = 2
dropout = 1.0

# Make variables
weights = tf.Variable(tf.truncated_normal([num_hidden,dict_length],stddev=1.0,dtype=tf.float32),name="weights1")
biases = tf.Variable(tf.truncated_normal([dict_length],stddev=1.0,dtype=tf.float32), name="biases1")

# Create input data
cell = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
X_reshape = tf.reshape(X_train,[batch_size, num_steps, dict_length])
X_unstack = tf.unstack(tf.to_float(X_reshape))
print X_reshape.shape

# Create multi-layer RNN
layer_cell=[]
for _ in range(num_layers):
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                          input_keep_prob=dropout,
                                          output_keep_prob=dropout)
    layer_cell.append(lstm_cell)

cell = tf.contrib.rnn.MultiRNNCell(layer_cell, state_is_tuple=True)
outputs, last_states = tf.contrib.rnn.static_rnn(
    cell,
    tf.unstack(tf.to_float(X_reshape)),dtype=tf.float32)

output_reshape=tf.reshape(outputs, [batch_size*num_steps,num_hidden])
print output_reshape.shape

pred=tf.matmul(output_reshape, weights) + biases
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y_train))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)


init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())    

plot_loss=[]

with tf.Session() as sess:
        
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)        
        for i in range(300):
            loss,_,y_target,y_pred,output=sess.run([cost,optimizer,Y_train,pred,outputs])
            plot_loss.append([loss])

            if i% 25 ==0:
                print("iteration: ",i," loss: ",loss)
                
        print(y_target)
        print(np.argmax(y_pred,1))          
        coord.request_stop()
        coord.join(threads)
        sess.close()    

#Lets look at one input data point at each step and its prediction
print y_pred.shape
print("Input Sentence")
sn=0 #The sentence number
print([word for word in sentence_1])
print("Predicted words")
print([int2word[ind] for ind in np.argmax(y_pred[sn::2],1)])
