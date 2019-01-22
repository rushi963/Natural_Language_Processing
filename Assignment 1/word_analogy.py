import os
import pickle
import numpy as np
from scipy import spatial
model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

file_in = open('word_analogy_test.txt', 'r')
#file_out = open('word_analogy_cross_entropy.txt', 'w')
file_out = open('word_analogy_test_predictions_nce.txt', 'w')
output = ""

for line in file_in:
    relation_scores = []
    option_scores = []
    relation_average = []
    # Taking the relations pairs
    relation_pairs = line.strip().split("||")[0]
    relations = relation_pairs.strip().split(",")
    # Going through each relation pair
    for pair1 in relations:
        one, two = pair1.strip().split(":")
        embedding_one = embeddings[dictionary[one[1:]]]
        embedding_two = embeddings[dictionary[two[:-1]]]
        # Calculating the difference vector for each relation pair
        relations_difference = np.subtract(embedding_one, embedding_two)
        relation_scores.append(relations_difference.tolist())
    # Computing the average difference vector for the relation pairs
    relation_average = np.mean(relation_scores, axis=0)

    # Taking the option pairs
    option_pairs = line.strip().split("||")[1]
    options = option_pairs.strip().split(",")
    # Going through each option pair
    for pair2 in options:
        first, second = pair2.strip().split(":")
        embedding_first = embeddings[dictionary[first[1:]]]
        embedding_second = embeddings[dictionary[second[:-1]]]
        # Calculating the difference vector for each option pair
        options_difference = np.subtract(embedding_first, embedding_second)
        option_scores.append(options_difference)

    # Computing cosine similarity between relation difference vector and each option difference vector
    result = []
    for option in option_scores:
        result.append(1 - spatial.distance.cosine(option, relation_average))

    # Taking the least and most illustrative option pair
    least = result.index(np.min(result))
    most = result.index(np.max(result))

    output += option_pairs.strip().replace(",", " ") + " " + options[least] + " " + options[most] + "\n"

file_out.write(output)
file_out.close()