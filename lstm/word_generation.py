# this is jupyter notebook lab from Nvidia to predict next word using LSTM

import numpy as np #numpy is "numerical python" and is used in deep learning mostly for its n-dimensional array
import tensorflow as tf
from tensorflow.python.framework import ops

small_dict=['EOS','a','my','sleeps','on','dog','cat','the','bed','floor'] #'EOS' means end of sentence.
X=np.array([[2,6,3,4,2,8,0],[1,5,3,4,7,9,0]],dtype=np.int32)
print([small_dict[ind] for ind in X[1,:]]) #Feel free to change 1 to 0 to see the other sentence.


plot_loss=[]
num_hidden=24
num_steps=X.shape[1]
dict_length=len(small_dict)
batch_size=2
tf.reset_default_graph()

## Make Variables
weights1 = tf.Variable(tf.truncated_normal([num_hidden,dict_length],stddev=1.0,dtype=tf.float32),name="weights1")
biases1 = tf.Variable(tf.truncated_normal([dict_length],stddev=1.0,dtype=tf.float32), name="biases1")

# Create input data
X_one_hot=tf.nn.embedding_lookup(np.identity(dict_length), X) #[batch,num_steps,dictionary_length][2,6,7]
#print X_one_hot
print "X_one_hot " + str(X_one_hot.shape)
y=np.zeros((batch_size,num_steps),dtype=np.int32)
print y.shape
print y
y[:,:-1]=X[:,1:]

print X
print y
y_one_hot=tf.unstack(tf.nn.embedding_lookup(np.identity(dict_length), y),num_steps,1) #[batch,num_steps,dictionary_length][2,6,7]

y_target_reshape=tf.reshape(y_one_hot,[batch_size*num_steps,dict_length])
print tf.Session().run(y_target_reshape)
print y_target_reshape.shape

#Create our LSTM
cell = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)

X_unstack = tf.unstack(tf.to_float(X_one_hot))
print X_unstack.shape

outputs, last_states = tf.contrib.rnn.static_rnn(
    cell=cell,
    dtype=tf.float32,
    inputs=tf.unstack(tf.to_float(X_one_hot),num_steps,1))

output_reshape=tf.reshape(outputs, [batch_size*num_steps,num_hidden])
print output_reshape.shape
