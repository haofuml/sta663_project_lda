import numpy as np

# visualization of LDA, gamma.shape = vocab_size, topic_num  
def topic_viz(gamma, vocab, topk=10):
    sorted_idx = np.argsort(-gamma, axis=0)
    sorted_gamma = -np.sort(-gamma, axis=0)
    sorted_gamma /= np.sum(sorted_gamma, axis=0)
    _,topic_num = gamma.shape
    for i in range(topic_num):
        print('topic %i:'%(i+1))
        print('top-%i key words:\n'%topk, vocab[sorted_idx[:topk,i]])
        print('distribution of top-10 key words:\n',sorted_gamma[:topk,i])