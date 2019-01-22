import os
import pickle
from scipy import spatial

model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

# Loading the best model
dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

# Retrieve the word by using its word id
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

# Converting the word embeddings to tuple
embeddings_tuple = [tuple(i) for i in embeddings]

# Creating a dictionary that returns the word id for a given embedding
reverse_embeddings = {}
for j in range(len(embeddings_tuple)):
    reverse_embeddings[embeddings_tuple[j]] = j

# Words given to us
words = ["first", "american", "would"]

# Iterate for each word
for word in words:
    # Get the word id given the word
    word_id = dictionary[word]
    dict = {}
    # Calculating the cosine similarity between the words in the vocabulary with the given word
    for i in range(len(embeddings)):
        dict[embeddings_tuple[i]] = 1 - spatial.distance.cosine(embeddings[word_id], embeddings[i])

    # Sort all the embeddings in decreasing order based on the cosine similarity values
    sorted_embeddings = sorted(dict.items(), key=lambda kv: kv[1], reverse=True)

    # We take top 21 words because the first word would be the given word itself
    top_20_words = []
    for i in range(1, 21):
        top_word_embedding = sorted_embeddings[i][0]
        # Get word id from its
        top_word_id = reverse_embeddings[top_word_embedding]
        top_word = reverse_dictionary[top_word_id]
        top_20_words.append(top_word)
    print(word, top_20_words)