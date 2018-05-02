"""
Collapsed Gibbs Sampling Implementation of LDA
"""
import numpy as np
import sys
import random
from scipy.special import gamma, gammaln, psi
from scipy.stats import *
from scipy import *
import argparse
from sta663_project_lda.visualization.demo_topics import topic_viz

class LDAGibbs(object):
    def __init__(self, data_path, ntopics):
        self.TOPICS = ntopics
        self.alpha = np.ones(self.TOPICS)
        for i in range(self.TOPICS):
            self.alpha[i] = 0.1
        self.beta = 0.01
        data = np.load(data_path).T
        self.DOCS = data.shape[0]
        self.VOCABS = data.shape[1]
        self.documents = {}
        for i, doc in enumerate(data):
            tmp_doc = []
            for j, word in enumerate(doc):
                if(word==0):
                    continue
                while(word!=0):
                    tmp_doc.append(j)
                    word -=1
            random.shuffle(tmp_doc)
            self.documents[i] = tmp_doc


        self.theta = np.zeros([self.DOCS, self.TOPICS])
        self.phi = np.zeros([self.TOPICS, self.VOCABS])

        self.sample_theta = np.zeros([self.DOCS, self.TOPICS])
        self.sample_phi = np.zeros([self.TOPICS, self.VOCABS])

    def Loss(self):
        ll = 0
        for z in range(self.TOPICS): # Symmetric Dirichlet Distribution (beta distribution in high dimension) Words | Topics, beta
            ll += gammaln(self.VOCABS*self.beta) # gamma distribution
            ll -= self.VOCABS * gammaln(self.beta)
            ll += np.sum(gammaln(self.cntTW[z] + self.beta))
            ll -= gammaln(np.sum(self.cntTW[z] + self.beta))
        for doc_num, doc in enumerate(self.documents): # Dirichlet Distribution: Topics | Docs, alpha
            ll += gammaln(np.sum(self.alpha)) # Beta(alpha)
            ll -= np.sum(gammaln(self.alpha))
            ll += np.sum(gammaln(self.cntDT[doc_num] + self.alpha))
            ll -= gammaln(np.sum(self.cntDT[doc_num] + self.alpha))
        return ll

    def gibbs_update(self, d, w, pos):
        z = self.topicAssignments[d][pos] # old theme
        self.cntTW[z,w] -= 1
        self.cntDT[d,z] -= 1
        self.cntT[z] -= 1

        prL = (self.cntDT[d] + self.alpha) / (self.lenD[d] -1 + np.sum(self.alpha))
        prR = (self.cntTW[:,w] + self.beta) / (self.cntT + self.beta * self.VOCABS)
        prFullCond = prL * prR
        prFullCond /= np.sum(prFullCond)
        new_z = np.random.multinomial(1, prFullCond).argmax()
        self.topicAssignments[d][pos] = new_z
        self.cntTW[new_z, w] += 1
        self.cntDT[d, new_z] += 1
        self.cntT[new_z] += 1

    def update_alpha_beta(self):

        # Update Beta
        x = 0
        y = 0
        for z in range(self.TOPICS):
            x += np.sum(psi(self.cntTW[z] + self.beta) - psi(self.beta))
            y += psi(np.sum(self.cntTW[z] + self.beta)) - psi(self.VOCABS * self.beta)
        self.beta = (self.beta * x) / (self.VOCABS * y)       # UPDATE BETA

        # Update Alpha
        x = 0
        y = 0
        for d in range(self.DOCS):
            y += psi(np.sum(self.cntDT[d] + self.alpha)) - psi(np.sum(self.alpha))
            x += psi(self.cntDT[d] + self.alpha) - psi(self.alpha)
        self.alpha *= x / y                       # UPDATE ALPHA

    def update_phi_theta(self):
        for d in range(self.DOCS):
            for z in range(self.TOPICS):
                self.sample_theta[d][z] = (self.cntDT[d][z] + self.alpha[z])/ (self.lenD[d] + np.sum(self.alpha))
        for z in range(self.TOPICS):
            for w in range(self.VOCABS):
                self.sample_phi[z][w] = (self.cntTW[z][w] + self.beta) / (self.cntT[z] + self.beta * self.VOCABS)

    def print_alpha_beta(self):
        print('Alpha')
        for i in range(self.TOPICS):
            print(self.alpha[i])
        print('Beta: {}'.format(beta))


    def run(self,max_iter = 50):
        burnin = max_iter*0.8
        self.topicAssignments = {}
        self.cntTW = np.zeros([self.TOPICS, self.VOCABS]) # count topic to words
        self.cntDT = np.zeros([self.DOCS, self.TOPICS]) # count docs to topics
        self.cntT = np.zeros(self.TOPICS)
        self.lenD = np.zeros(self.DOCS)

        # Iterate All the Documents, Initilaze the probibability matrix
        for doc_num,doc in enumerate(self.documents):
            doc_size = len(self.documents[doc])
            tmp = np.random.randint(0,self.TOPICS, size = doc_size)
            self.topicAssignments[doc_num] = tmp
            for i, word in enumerate(self.documents[doc]):
                self.cntTW[tmp[i],word] += 1
                self.cntDT[doc_num, tmp[i]] += 1
                self.cntT[tmp[i]] += 1
                self.lenD[word] +=1


        print('LIKELIHOOD:\n', self.Loss())
        self.print_alpha_beta()
        SAMPLES = 0
        for s in range(max_iter):
            print('Iter: {}'.format(s))
            for doc_num, doc in enumerate(self.documents):
                for i, word in enumerate(self.documents[doc]):
                    self.gibbs_update(doc_num, word, i) # word itself is its numerate.
            self.update_alpha_beta()
            print('Loss{}'.format(self.Loss()))
            self.print_alpha_beta()

            if(s>burnin):
                self.update_phi_theta()
                self.theta +=self.sample_theta
                self.phi += self.sample_phi

        self.theta /= (max_iter - burnin-1)
        self.phi /= (max_iter - burnin-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gibbs Sampling Training Paramters.')
    parser.add_argument('-K', dest='K', type=int,
                            help='number of Topics', default=2)
    parser.add_argument('--datadir', dest='datadir', action='store',
                            help='Path of the genearted data', default='./data/toydata_mat.npy')
    args = parser.parse_args()
    lda = LDAGibbs(args.datadir, args.K)
    lda.run()
    # print("Theta: {}".format(lda.theta))

    vocabulary = np.load('./data/toydata_voc.npy')
    topic_viz(lda.phi.T,vocabulary,topk=5)

