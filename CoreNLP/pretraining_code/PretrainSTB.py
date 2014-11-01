import numpy as np
import numpy.random as npr
import NLMLayers as nlml
import NLModels as nlm
import cPickle as pickle
from HelperFuncs import zeros, ones, randn, rand_word_seqs
import CorpusUtils as cu

#######################
# Test scripting code #
#######################

def some_nearest_words(keys_to_words, sample_count, W1=None, W2=None):
    assert(not (W1 is None))
    if not (W2 is None):
        W = np.hstack((W1, W2))
    else:
        W = W1
    norms = np.sqrt(np.sum(W**2.0,axis=1,keepdims=1))
    W = W / (norms + 1e-5)
    max_valid_key = np.max(keys_to_words.keys())
    W = W[0:(max_valid_key+1),:]
    # 
    source_keys = np.zeros((sample_count,)).astype(np.uint32)
    neighbor_keys = np.zeros((sample_count, 10)).astype(np.uint32)
    all_keys = np.asarray(keys_to_words.keys()).astype(np.uint32)
    for s in range(sample_count):
        i = npr.randint(0,all_keys.size)
        source_k = all_keys[i]
        neg_cos_sims = -1.0 * np.sum(W * W[source_k], axis=1)
        sorted_k = np.argsort(neg_cos_sims)
        source_keys[s] = source_k
        neighbor_keys[s,:] = sorted_k[1:11]
    source_words = []
    neighbor_words = []
    for s in range(sample_count):
        source_words.append(keys_to_words[source_keys[s]])
        neighbor_words.append([keys_to_words[k] for k in neighbor_keys[s]])
    return [source_keys, neighbor_keys, source_words, neighbor_words]

def record_word_vectors(w2k, cam, file_name):
    """Write some trained word vectors to the given file."""
    wv_file = open(file_name, 'w')
    all_words = w2k.keys()
    all_words.sort()
    wv_std_dev = 0.0
    for word in all_words:
        key = w2k[word]
        word_vec = cam.word_layer.params['W'][key]
        wv_std_dev += np.mean(word_vec**2.0)
        word_vals = [word]
        word_vals.extend([str(val) for val in word_vec])
        wv_file.write(" ".join(word_vals))
        wv_file.write("\n")
    wv_file.close()
    wv_std_dev = np.sqrt(wv_std_dev / len(all_words))
    print("approximate word vector std-dev: {0:.4f}".format(wv_std_dev))
    return

def init_biases_with_lups(cam, w2lup, w2k):
    """Init class layer biases in cam with log unigram probabilities."""
    for w in w2lup:
        cam.class_layer.params['b'][w2k[w]] = max(w2lup[w], -6.0)
    return


if __name__=="__main__":
    # select source of phrases to pre-train word vectors for.
    data_dir = './training_text/train_and_dev' # TO USE TRAIN AND DEV SETS
    #data_dir = './training_text/train_only' # TO USE ONLY TRAIN SET
    # set some parameters.
    min_count = 2 # lower-bound on frequency of words in kept vocab
    sg_window = 5 # context size or skip-gram sampling
    ns_count = 15 # number of negative samples for negative sampling
    wv_dim = 70   # dimensionality of vectors to pre-train
    cv_dim = 10   # this won't be used. it's safe to ignore
    lam_l2 = 0.5 * wv_dim**0.5 # will be used to constrain vector norms

    # generate the training vocabulary
    sentences = cu.SentenceFileIterator(data_dir)
    key_dicts = cu.build_vocab(sentences, min_count=2, compute_hs_tree=True, \
                            compute_ns_table=True, down_sample=0.0)
    w2k = key_dicts['words_to_keys']
    k2w = key_dicts['keys_to_words']
    w2lups = key_dicts['word_log_probs']
    neg_table = key_dicts['ns_table']
    unk_word = key_dicts['unk_word']
    hsm_code_dict = key_dicts['hs_tree']
    sentences = cu.SentenceFileIterator(data_dir)
    tr_phrases = cu.sample_phrases(sentences, w2k, unk_word=unk_word, \
                                max_phrases=100000)
    # get some important properties of the generated training vocabulary
    max_cv_key = len(tr_phrases) + 1
    max_wv_key = max(w2k.values())
    max_hs_key = key_dicts['hs_tree']['max_code_key']


    # initialize the model to-be-trained
    cam = nlm.CAModel(wv_dim, cv_dim, max_wv_key, max_cv_key, \
                use_ns=True, max_hs_key=max_hs_key, \
                lam_wv=lam_l2, lam_cv=lam_l2, lam_cl=lam_l2)
    # init parameters in word, context, and classification layers
    cam.use_tanh = True
    cam.init_params(0.02)
    # set parameters in context layer to 0s, across the board
    cam.context_layer.init_params(0.0)
    # tell the model to train subject to dropout and weight fuzzing
    cam.set_noise(drop_rate=0.5, fuzz_scale=0.02)
    # init prediction layer biases with log unigram probabilities
    init_biases_with_lups(cam, w2lups, w2k)
    # NOTE: given the properties of negative sampling, initializing with the
    # log unigram probabilities is actually kind of silly. But, we'll leave it
    # in there because I didn't know better at the time, and the resulting
    # vectors performed adequately.

    # initialize samplers for drawing positive pairs and negative contrastors
    pos_sampler = cu.PhraseSampler(tr_phrases, sg_window)
    neg_sampler = cu.NegSampler(neg_table=neg_table, neg_count=ns_count)

    # train all parameters using the training set phrases
    learn_rate = 1e-2
    decay_rate = 0.975
    for i in range(50):
        cam.train(pos_sampler, neg_sampler, 250, 50001, train_ctx=False, \
                  train_lut=True, train_cls=True, learn_rate=learn_rate)
        learn_rate = learn_rate * decay_rate
        record_word_vectors(w2k, cam, "wv_d{0:d}_mc{1:d}.txt".format(wv_dim, min_count))
        [s_keys, n_keys, s_words, n_words] = some_nearest_words( k2w, 10, \
                  W1=cam.word_layer.params['W'], W2=None)
        for w in range(10):
            print("{0:s}: {1:s}".format(s_words[w],", ".join(n_words[w])))







##############
# EYE BUFFER #
##############
