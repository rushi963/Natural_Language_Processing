import tensorflow as tf


def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    # Calculating A = log(exp({u_o}^T v_c)) = {u_o}^T v_c
    A = tf.diag_part(tf.matmul(inputs, tf.transpose(true_w)))

    # Calculating B = log(\sum{exp({u_w}^T v_c)})
    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, tf.transpose(true_w))), [1]))

    return tf.subtract(B, A)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weights: Weights for nce loss. Dimension is [Vocabulary, embedding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimension is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimension is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    # Extracting batch size, embedding size and sample size
    batch_size = inputs.get_shape().as_list()[0]
    embedding_size = inputs.get_shape().as_list()[1]
    sample_size = len(sample)

    # Defining small epsilon values which are added before taking log
    e1 = tf.constant(1e-10, shape=[batch_size, 1])
    e2 = tf.constant(1e-10, shape=[batch_size, sample_size])

    ##### Part 1 #####(Calculating log(Pr(D=1,w_o|w_c)))

    # Extracting label weights using embedding_lookup
    predicting_embeddings = tf.reshape(tf.nn.embedding_lookup(weights, labels), [batch_size, embedding_size])

    # Extracting label bias using embedding_lookup
    predicting_bias = tf.nn.embedding_lookup(biases, labels)
    #print(predicting_embeddings, predicting_bias)

    # Converting unigram_prob to tensor
    unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

    # Extracting unigram probability of labels using embedding_lookup
    unigram_labels = tf.nn.embedding_lookup(unigram_prob, labels)
    #print(unigram_prob, unigram_labels)

    # Calculating s(w0,wc)
    context = tf.add(tf.reshape(tf.diag_part(tf.matmul(inputs, tf.transpose(predicting_embeddings))), [batch_size, 1]), predicting_bias)

    # Calculating log[kPr(w_o)]
    unigram_labels = tf.log(tf.scalar_mul(sample_size, unigram_labels))
    #print(context, unigram_labels)

    # Calculating log(Pr(D=1, w_o | w_c)) and adding a small value to address issue of log(0)
    predicting_words = tf.log(tf.add(tf.sigmoid(tf.subtract(context, unigram_labels)), e1))
    #print(predicting_words)

    ##### Part 2 ##### log(1 - Pr(D=1,w_x|w_c))

    # Converting sample to tensor
    sample = tf.convert_to_tensor(sample, dtype=tf.int32)

    # Extracting negative sample weights using embedding_lookup
    negative_embeddings = tf.nn.embedding_lookup(weights, sample)

    # Extracting unigram_prob of negative samples using embedding_lookup
    unigram_negative = tf.transpose(tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample), [sample_size, 1]))
    #print(negative_embeddings, unigram_negative)

    # Extracting negative sample bias using embedding_lookup
    negative_bias = tf.transpose(tf.reshape(tf.nn.embedding_lookup(biases, sample), [sample_size, 1]))

    # Using tile to convert from (sample_size x 1) to (batch_size x sample_size) by copying values across columns
    negative_bias = tf.tile(negative_bias, [batch_size, 1])
    #print(negative_bias)

    # Calculating s(wx,wc)
    negative = tf.add(tf.matmul(inputs, tf.transpose(negative_embeddings)), negative_bias)
    #print(negative)

    # Calculating log[kPr(w_x)]
    unigram_negative = tf.log(tf.scalar_mul(sample_size, unigram_negative))

    # Using tile to convert from (sample_size x 1) to (batch_size x sample_size) by copying values across columns
    unigram_negative = tf.tile(unigram_negative, [batch_size, 1])
    #print(unigram_negative)

    # Creating matrix of size (batch_size x sample_size) and converting to tensor
    one = [[1.0 for j in range(sample_size)] for i in range(batch_size)]
    ones = tf.convert_to_tensor(one, dtype=tf.float32)

    # Calculating log(1-Pr(D=1,w_x|w_c))
    negative_words = tf.subtract(negative, unigram_negative)
    negative_words = tf.subtract(ones, tf.sigmoid(negative_words))
    negative_words = tf.reshape(tf.reduce_sum(tf.log(tf.add(negative_words, e2)), [1]), [batch_size, 1])
    #print(negative_words)

    # Calculating -1 x (log(Pr(D=1,w_o|w_c)) - summation of log(1 - Pr(D=1,w_x|w_c)))
    nce = tf.scalar_mul(-1, tf.add(predicting_words, negative_words))
    #print(nce)

    return nce