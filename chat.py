
from io import open
import tensorflow as tf
from nltk import word_tokenize
import pickle
import re
import numpy as np
import string

# retrive files
with open('word_to_int.pkl', 'rb') as f:
	word_to_int = pickle.load(f)
with open('int_to_word.pkl', 'rb') as f:
	int_to_word = pickle.load(f) 


def clean_text(text):
    # clean the conractions, clean starting symbols '>' and other symbols
    text = text.lower()
    
    # replace contractions
    text = word_tokenize(text)
    new_text = []
    for word in text:
        if word == '>':
            continue
#         if word in contractions:
#             word = contractions[word]
#             new_text += word.split(' ')
#             continue
        new_text.append(word)
    
    text = " ".join(new_text)
    
    # everything in the brackets
    text = re.sub(r'[_"\-%()|+&=*%#$@\[\]/]', ' ', text)
    # <br/> 
    # text = re.sub(r'<br />', ' ', text)
    # text = re.sub(r'\'', ' ', text)
    return text


def text_to_seq(text):
    '''Prepare the text for the model'''
    
    text = clean_text(text)
    return [word_to_int.get(word, word_to_int['<UNK>']) for word in word_tokenize(text)]


def chat():
	# Create your own review or use one from the dataset
	pad = word_to_int["<PAD>"] 
	batch_size = 32

	checkpoint = "./best_model.ckpt"

	loaded_graph = tf.Graph()
	with tf.Session(graph=loaded_graph) as sess:
	    # Load saved model
	    loader = tf.train.import_meta_graph(checkpoint + '.meta')
	    loader.restore(sess, checkpoint)

	    input_data = loaded_graph.get_tensor_by_name('input:0')
	    logits = loaded_graph.get_tensor_by_name('predictions:0')
	    X_length = loaded_graph.get_tensor_by_name('X_length:0')
	    y_length = loaded_graph.get_tensor_by_name('y_length:0')
	    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
	    
	    while True:  	
		    input_sentence = raw_input('> ')
		    text = text_to_seq(input_sentence)
		    #Multiply by batch_size to match the model's input parameters
		    answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
		                                      y_length: [np.random.randint(35,40)], 
		                                      X_length: [len(text)]*batch_size,
		                                      keep_prob: 1.0})[0]
		    tokens = [int_to_word[i] for i in answer_logits if i != pad]
		    response = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
		    print('  Response: {}'.format(response))

if __name__ == '__main__':
	chat()
