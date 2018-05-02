import argparse
import numpy as np
from sta663_project_lda.visualization.demo_topics import topic_viz


class LDASVI(object):
    
    def __init__(self,  datadir, K, alpha0=None, gamma0=None, 
        MB=256, kappa=0.5, tau0=256, eps=1e-3):

        self.wordcnt_mat = np.load(datadir) # word-count matrix
        self.vocab_size = self.wordcnt_mat.shape[0] # number of words in vocabulary
        self.D = self.wordcnt_mat.shape[1] # number of documents
        self.K = K # number of topics

        self.alpha0 = 1/self.K if alpha0==None else alpha0
        self.gamma0 = 1/self.K if gamma0==None else gamma0
        self.MB = MB  # mini-batch size
        self.epoch_len = self.D // self.MB # number of iterations in each epoch
        self.kappa = kappa  # learning rate parameters
        self.tau0 = tau0  # learningg rate parameters
        self.eps = eps  # criterion of convergence for local updates
    
    def train_numpy(self, epoch=10, seed=0, printcycle=10):
        import numpy as np
        from scipy.special import psi

        np.random.seed(seed)
        gamma = np.random.rand(self.vocab_size, self.K) # initialization of topics
        phi = {} # topic assignments
        
        for ep in range(epoch):
            order = np.random.permutation(self.D)
            for t in range(self.epoch_len):
                itr = ep * self.epoch_len + t
                if printcycle!=0 and itr%printcycle==0:
                    print('starting iteration %i'%itr)
                
                '''E-Step: update local variational parameters(phi,alpha) till convergent'''
                sample_id = order[t*self.MB:(t+1)*self.MB]
                alpha = np.ones((self.K, self.MB)) # initialization of topic proportions
                psi_sum_gam = psi(np.sum(gamma, axis=0))
                diff = self.eps + 1
                while(diff>self.eps):
                    diff = 0
                    for i in range(self.MB):
                        tmp_id = np.nonzero(self.wordcnt_mat[:, sample_id[i]])[0]
                        tmp_cnt = self.wordcnt_mat[tmp_id, sample_id[i]]
                        # update topic assignement for each word in each document
                        tmp_phi = np.exp(psi(gamma[tmp_id,:]) - psi_sum_gam + psi(alpha[:,i]) - psi(np.sum(alpha[:,i])))
                        phi[i] = tmp_phi / np.reshape(np.sum(tmp_phi,axis=1), (-1,1))
                        # update topic proportion for each document
                        tmp_alpha = self.alpha0 + tmp_cnt[None,:] @ phi[i]
                        # accumulate diff to decide local convergence
                        diff += np.sum(np.abs(tmp_alpha-alpha[:,i]))
                        alpha[:,i] = tmp_alpha
                    diff = diff / self.K / self.MB
                
                '''M-Step: update global variational parameters(gamma)'''
                tmp_gamma = np.zeros((self.vocab_size, self.K))                  
                for i in range(self.MB):
                    tmp_id = np.nonzero(self.wordcnt_mat[:, sample_id[i]])[0]
                    tmp_cnt = self.wordcnt_mat[tmp_id, sample_id[i]]
                    tmp_gamma[tmp_id, :] += phi[i] * tmp_cnt[:,None]
                tmp_gamma = self.gamma0 + tmp_gamma * self.D / self.MB
                rho_t = (self.tau0 + itr)**(-self.kappa)
                gamma = (1-rho_t)*gamma + rho_t*tmp_gamma    
        return gamma # no need to return alpha, since alpha only includes topic proportion of a mini-batch of documents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stochastic Variational Inference Training Paramters.')
    parser.add_argument('--datadir', dest='datadir', action='store',
                        help='Path of the training data', default='./data/toydata_mat.npy')
    parser.add_argument('-K', dest='K', type=int,
                        help='Number of Topics', default=2)
    parser.add_argument('--MB', dest='MB', type=int,
                        help='minibatch size', default=20)

    args = parser.parse_args()
    lda = LDASVI(args.datadir, args.K, MB=args.MB)
    gamma = lda.train_numpy(epoch=50)
    vocabulary = np.load('./data/toydata_voc.npy')
    topic_viz(gamma,vocabulary,topk=5)
