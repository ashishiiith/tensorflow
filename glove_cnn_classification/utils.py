
# this file is for pre=processing the data to generate vector of word ids.

import os
import sys
import tensorflow as tf
import numpy as np
#import tf.keras.preprocessing.text.Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

TEXT_DATA_DIR = sys.argv[1]
MAX_NB_WORDS = 20000 # we would be considering only top 20k words by frequency
MAX_SEQUENCE_LEN = 100

def to_one_hot(index, size):

    array = np.zeros(size)
    array[index] = 1
    return array

def preprocess_text():
	
    texts = []
    labels_index = {}
    labels = []
    label_index = 0
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
	    labels_index[name] = label_index
            for fname in sorted(os.listdir(path)):
 	        if fname.isdigit():
	            fpath = os.path.join(path, fname)
		    f = open(fpath)
		    t = f.read()
		    i = t.find('\n\n') #skip header
		    if i > 0:
		        t = t[i:]
		    texts.append(t)
		    f.close()
		    labels.append(label_index)
	label_index+=1

    print "number of texts found " + str(len(texts))
    print "number of labels " + str(len(labels_index))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    
    #print len(word_index)

    data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LEN)
    
    label = []
    
    for index in labels:
        label.append(to_one_hot(index, len(labels_index)))

    label = tf.convert_to_tensor(np.asarray(label), dtype=tf.float32)
    print label.shape
    print data.shape

    


preprocess_text()
