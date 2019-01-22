import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    # Taking the transpose - LxN (tags - rows, words - columns)
    emission = emission_scores.transpose()

    # Creating two matrices for storing scores and tags
    scores = np.zeros((L, N))
    tags = np.zeros((L, N), dtype=int)

    # Initialize first column using start scores
    for i in range(L):
        scores[i][0] = emission[i][0] + start_scores[i]

    tag_sequence = []
    for i in range(1, N):
        for j in range(L):
            score_temp = np.zeros(L)

            # Calculating scores and storing best scoring tag positions (used for backtracking)
            for k in range(L):
                score_temp[k] = scores[k][i - 1] + trans_scores[k][j]
            pos_max = score_temp.argmax()
            score_max = score_temp.max()
            tags[j][i] = pos_max
            scores[j][i] = score_max + emission[j][i]

    score_temp = np.zeros(L)

    # Adding the end transitions scores
    for i in range(L):
        scores[i][N - 1] += end_scores[i]
        score_temp[i] = scores[i][N - 1]
    final_score = score_temp.max()
    pos_max = score_temp.argmax()
    tag_sequence.append(pos_max)

    # Backtracking to find the best scoring tag sequence
    for i in range(N - 1, 0, -1):
        pos_max = tags[pos_max][i]
        tag_sequence.append(pos_max)
    tag_sequence.reverse()

    return final_score, tag_sequence