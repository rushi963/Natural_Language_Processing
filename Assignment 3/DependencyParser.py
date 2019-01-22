import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():

            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            # Fixed Embeddings
            #self.embeddings = tf.Variable(embedding_array, dtype=tf.float32, trainable=False)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
            # Creating placeholders for the train inputs, train labels and test inputs
            self.train_inputs = tf.placeholder(tf.int32, shape=(Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(tf.float32, shape=(Config.batch_size, parsing_system.numTransitions()))
            self.test_inputs = tf.placeholder(tf.int32)

            # Obtaining the word embeddings and reshaping for multiplication
            train_embedding = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embedding = tf.reshape(train_embedding, [Config.batch_size, Config.embedding_size * Config.n_Tokens])



            # Initializing the weights_input randomly from uniform distribution between [-1,1] as mentioned in the paper
            #weights_input = tf.Variable(tf.random_uniform([Config.embedding_size*Config.n_Tokens,Config.hidden_size],-1,1))

            # Initializing the weights_input randomly from normal distribution
            weights_input = tf.Variable(tf.random_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_size], mean=0.0, stddev=0.1))


            # Initialize the biases_input randomly from uniform distribution between [-1,1]
            #biases_input = tf.Variable(tf.random_uniform([Config.hidden_size],-1,1))

            # Initializing the biases_input with zeros
            biases_input = tf.Variable(tf.zeros([Config.hidden_size]))


            # Initializing the weights_output randomly from uniform distribution between [-1,1] as mentioned in the paper
            #weights_output = tf.Variable(tf.random_uniform([Config.hidden_size, parsing_system.numTransitions()],-1,1))

            # Initializing the weights_output randomly from normal distribution
            weights_output = tf.Variable(tf.random_normal([Config.hidden_size, parsing_system.numTransitions()], mean=0.0, stddev=0.1))


            # Using 3 hidden layers
            '''
            weights_input = {
                'h1': tf.Variable(tf.random_normal([Config.embedding_size * Config.n_Tokens, Config.layer1_size], mean=0.0, stddev=0.1)),
                'h2': tf.Variable(tf.random_normal([Config.layer1_size, Config.layer2_size], mean=0.0, stddev=0.1)),
                'h3': tf.Variable(tf.random_normal([Config.layer2_size, Config.layer3_size], mean=0.0, stddev=0.1))
            }
            biases_input = {
                'b1': tf.Variable(tf.zeros([Config.layer1_size])),
                'b2': tf.Variable(tf.zeros([Config.layer2_size])),
                'b3': tf.Variable(tf.zeros([Config.layer3_size]))
            }
            weights_output = tf.Variable(tf.random_normal([Config.layer3_size, parsing_system.numTransitions()], mean=0.0, stddev=0.1))
            '''

            # Using 2 hidden layers
            '''
            weights_input = {
                'h1': tf.Variable(tf.random_normal([Config.embedding_size * Config.n_Tokens, Config.layer1_size], mean=0.0, stddev=0.1)),
                'h2': tf.Variable(tf.random_normal([Config.layer1_size, Config.layer2_size], mean=0.0, stddev=0.1)),
            }
            biases_input = {
                'b1': tf.Variable(tf.zeros([Config.layer1_size])),
                'b2': tf.Variable(tf.zeros([Config.layer2_size])),
            }
            weights_output = tf.Variable(tf.random_normal([Config.layer2_size, parsing_system.numTransitions()], mean=0.0, stddev=0.1))
            '''

            # Using 3 separate parallel hidden layers
            '''
            weights_input = {
                'w_words': tf.Variable(tf.random_normal([Config.embedding_size * 18, Config.hidden_size], 
                                                    mean=0.0, stddev=0.1)),
                'w_tags': tf.Variable(tf.random_normal([Config.embedding_size * 18, Config.hidden_size], 
                                                    mean=0.0, stddev=0.1)),
                'w_labels': tf.Variable(tf.random_normal([Config.embedding_size * 12, Config.hidden_size], 
                                                    mean=0.0, stddev=0.1)),
            }
            biases_input = {
                'b_words': tf.Variable(tf.zeros([Config.hidden_size])),
                'b_tags': tf.Variable(tf.zeros([Config.hidden_size])),
                'b_labels': tf.Variable(tf.zeros([Config.hidden_size]))
            }
            weights_output = tf.Variable(tf.random_normal([3*Config.hidden_size, parsing_system.numTransitions()], mean=0.0, stddev=0.1))
            '''

            # Predictions
            self.predictions = self.forward_pass(train_embedding, weights_input, biases_input, weights_output)

            # loss function
            logit_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictions, labels=tf.argmax(self.train_labels, axis=1)))

            # Regularization for 1 hidden layer
            regularization = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_output))

            # Regularization for 3 hidden layers
            '''
            regularization = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(weights_input['h1']) + tf.nn.l2_loss(weights_input['h2'])
                                           + tf.nn.l2_loss(weights_input['h3']) + tf.nn.l2_loss(biases_input['b1']) + tf.nn.l2_loss(biases_input['b2'])
                                           + tf.nn.l2_loss(biases_input['b3']) + tf.nn.l2_loss(weights_output))
            '''

            # Regularization for 2 hidden layers
            '''
            regularization = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(weights_input['h1']) + tf.nn.l2_loss(weights_input['h2'])
                         + tf.nn.l2_loss(biases_input['b1']) + tf.nn.l2_loss(biases_input['b2']) + tf.nn.l2_loss(weights_output))
            '''

            # Regularization for three separate parallel hidden layers
            '''
            regularization = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(weights_input['w_words'])+ tf.nn.l2_loss(weights_input['w_tags'])
                                         + tf.nn.l2_loss(weights_input['w_labels']) + tf.nn.l2_loss(biases_input['b_words']) + tf.nn.l2_loss(biases_input['b_tags'])
                                         + tf.nn.l2_loss(biases_input['b_labels']) + tf.nn.l2_loss(weights_output))
            '''

            # Loss + Regularization
            self.loss = logit_loss + regularization

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            #self.app = optimizer.apply_gradients(grads)

            # Gradient Clipping
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print(result)

        print("Train Finished.")

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print("Starting to predict on test set")
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print("Saved the test results.")
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        # 3 hidden layers
        '''
        h_layer1 = tf.pow(tf.add(tf.matmul(embed, weights_input['h1']), biases_input['b1']), 3)
        h_layer2 = tf.nn.relu(tf.add(tf.matmul(h_layer1, weights_input['h2']), biases_input['b2']))
        #h_layer2 = tf.pow(tf.add(tf.matmul(h_layer1, weights_input['h2']), biases_input['b2']), 3)
        h_layer3 = tf.nn.relu(tf.add(tf.matmul(h_layer2, weights_input['h3']), biases_input['b3']))
        #h_layer3 = tf.pow(tf.add(tf.matmul(h_layer2, weights_input['h3']), biases_input['b3']), 3)
        p = tf.matmul(h_layer3, weights_output)
        '''

        # 2 hidden layers
        '''
        h_layer1 = tf.pow(tf.add(tf.matmul(embed, weights_input['h1']), biases_input['b1']), 3)
        h_layer2 = tf.nn.relu(tf.add(tf.matmul(h_layer1, weights_input['h2']), biases_input['b2']))
        #h_layer2 = tf.pow(tf.add(tf.matmul(h_layer1, weights_input['h2']), biases_input['b2']), 3)
        p = tf.matmul(h_layer2, weights_output)
        '''

        # 3 parallel hidden layers
        '''
        word_index = Config.embedding_size*18
        tag_index = word_index + (Config.embedding_size*18)
        label_index = tag_index + (Config.embedding_size*12)
        
        word_embeddings = embed[:, 0: word_index]
        tags_embeddings = embed[:, word_index: tag_index]
        label_embeddings = embed[:, tag_index: label_index]
        
        h_words = tf.pow(tf.add(tf.matmul(word_embeddings, weights_input['w_words']), biases_input['b_words']), 3)
        h_tags = tf.pow(tf.add(tf.matmul(tags_embeddings, weights_input['w_tags']), biases_input['b_tags']), 3)
        h_labels = tf.pow(tf.add(tf.matmul(label_embeddings, weights_input['w_labels']), biases_input['b_labels']), 3)
        
        h = tf.concat([h_words, h_tags, h_labels], 1)
        p = tf.matmul(h, weights_output)
        '''

        # cubic activation function
        h = tf.pow(tf.add(tf.matmul(embed, weights_input), biases_input), 3)

        # sigmoid activation function
        #h = tf.nn.sigmoid(tf.add(tf.matmul(embed, weights_input), biases_input))

        # tanh activation function
        #h = tf.nn.tanh(tf.add(tf.matmul(embed, weights_input), biases_input))

        # relu activation function
        #h = tf.nn.relu(tf.add(tf.matmul(embed, weights_input), biases_input))

        p = tf.matmul(h, weights_output)

        return p


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    # For storing words, tags and labels
    s_words = []
    s_tags = []
    s_labels = []

    # Storing the top 3 elements of stack and buffer with their pos tags
    for i in range(0, 3):
        s_words.append(getWordID(c.getWord(c.getStack(i))))
        s_words.append(getWordID(c.getWord(c.getBuffer(i))))

        s_tags.append(getPosID(c.getPOS(c.getStack(i))))
        s_tags.append(getPosID(c.getPOS(c.getBuffer(i))))

    # Storing first leftmost and rightmost children of top 2 words in the stack with their pos tags and label tags
    # Storing second leftmost and rightmost children of top 2 words in the stack with their pos tags and label tags
    for i in range(0, 2):
        for j in range(0, 2):
            left_child = c.getLeftChild(c.getStack(i), j + 1)
            right_child = c.getRightChild(c.getStack(i), j + 1)

            s_words.append(getWordID(c.getWord(left_child)))
            s_words.append(getWordID(c.getWord(right_child)))

            s_tags.append(getPosID(c.getPOS(left_child)))
            s_tags.append(getPosID(c.getPOS(right_child)))

            s_labels.append(getLabelID(c.getLabel(left_child)))
            s_labels.append(getLabelID(c.getLabel(right_child)))

    # Storing leftmost of leftmost and rightmost of rightmost children of the top 2 words in the stack
    for i in range(0, 2):
        left_left_child = c.getLeftChild(c.getLeftChild(c.getStack(i), 1), 1)
        right_right_child = c.getRightChild(c.getRightChild(c.getStack(i), 1), 1)

        s_words.append(getWordID(c.getWord(left_left_child)))
        s_words.append(getWordID(c.getWord(right_right_child)))

        s_tags.append(getPosID(c.getPOS(left_left_child)))
        s_tags.append(getPosID(c.getPOS(right_right_child)))

        s_labels.append(getLabelID(c.getLabel(left_left_child)))
        s_labels.append(getLabelID(c.getLabel(right_right_child)))

    features = []
    features.extend(s_words)
    features.extend(s_tags)
    features.extend(s_labels)

    return features


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print(i, label)
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print("Found embeddings: ", foundEmbed, "/", len(knownWords))

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print(parsing_system.rootLabel)

    print("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print("Done.")

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

