#!/bin/python

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict

# functions for stemming and lemmatizing
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

brown_cluster = defaultdict(list)


def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """

    # Used a file containing the words and its corresponding cluster
    # Reference - http://www.cs.cmu.edu/~ark/TweetNLP/
    filename = open("100kpaths.txt")
    for line in filename:
        cluster = line.split('\t')
        if len(cluster) == 3:
            brown_cluster[cluster[1].strip()] = cluster[0].strip()


def token2features(sent, i, add_neighs=True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []

    # Basic Features

    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent) - 1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # Set 1

    # To check for url
    if word.startswith("http") or word.endswith((".com", ".org", ".net")):
        ftrs.append("IS_URL")
    # To check for '#'
    if word.startswith("#"):
        ftrs.append("IS_HASHTAG")
    # To check for '@'
    if word.startswith("@"):
        ftrs.append("IS_MENTION")

    # Set 2

    import string
    if word in string.punctuation:
        ftrs.append("IS_PUNCTUATION")

    # To check if the word is a probable adverb
    if word.endswith("ly"):
        ftrs.append("IS_ADVERB")

    # To check if the word is a probable verb
    if word.endswith("ed") or word.endswith("ing"):
        ftrs.append("IS_VERB")

    # To check if the word is a probable adjective
    if word.startswith("un") or word.endswith("st"):
        ftrs.append("IS_ADJECTIVE")

    # To check if the word contains hyphen
    if "-" in word:
        ftrs.append("IS_HYPHEN")

    # To check if the word is plural
    if word[-1:] is "s":
        ftrs.append("IS_PLURAL")

    # Set 3

    # Adding prefix and suffix of the word
    if len(word) > 3:
        ftrs.append("PREFIX=" + word[:3])
        ftrs.append("SUFFIX=" + word[-3:])
    elif len(word) == 3:
        ftrs.append("PREFIX=" + word[:2])
        ftrs.append("SUFFIX=" + word[-2])

    # Set 4

    # Adding stemmed and lemmatized words
    #ftrs.append("LEMMATIZED=" + lemmatizer.lemmatize(word.lower()))
    #ftrs.append("STEMMED=" + porter.stem(word))

    # Set 5

    # Adding cluster number as a feature(Brown Clustering)
    for k, v in brown_cluster.items():
        if word.encode('ascii', 'ignore') == k:
            ftrs.append("CLUSTER_" + v)

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i - 1, add_neighs=False):
                ftrs.append("PREV_" + pf)
        if i < len(sent) - 1:
            for pf in token2features(sent, i + 1, add_neighs=False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs


if __name__ == "__main__":
    sents = [
        ["I", "love", "food"]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)