import numpy as np
import tensorflow as tf

corpus = "He is the king. The king is royal. She is the royal queen"

corpus = corpus.lower()

word2int = {} #mapping of word to integer
int2word = {} #mapping of integer to word
index = 0

for word in corpus.split():
    if word!='.' and not word2int.has_key(word):
	word2int[word] = index
	int2word[index] = word 
	index+=1

print word2int['queen']
print int2word[0]

