import numpy as np
import argparse

def toydata_generator(doc_num=100, outdir='./data/'):
    """toy data generation"""
    
    xi=10
    a=(0.5,0.5) # sparse concentration parameters
    topics=['sports','science']

    wordMat=[]
    wordMat.append(['athlete','running','soccer','goal','touchdown'])
    wordMat.append(['telescope','space','beaker','goggles','testtube'])
    probMat=[]
    probMat.append([.5,.3,.1,.05,.05]) # beta_0: words distribution on the topic 0
    probMat.append([.2,.2,.2,.2,.2]) # beta_1: words distribution on the topic 1
    probMat=np.asarray(probMat)

    def generateDocword(wordMatrix,probMatrix,myxi,mya):
        mywords=[]
        theta=np.random.dirichlet(mya)
        N=np.random.poisson(myxi)
        for i in range(0,N):
            topInd=np.random.multinomial(1,theta)
            topInd=np.nonzero(topInd)[0][0]
            wordInd=np.random.multinomial(1,probMatrix[topInd,:])
            wordInd=np.nonzero(wordInd)[0][0]
            mywords.append(wordMatrix[topInd][wordInd])
        return mywords

    docs=[]
    for i in range(0, doc_num):
        docs.append(generateDocword(wordMat,probMat,xi,a))

    print('This dataset include two themes and ten words\n')
    print('The first topic is {} '.format(topics[0]))
    print('The second topic is {}\n'.format(topics[1]))
    print('{} documents are genearated.\n'.format(len(docs)))
    print('Toy data exmample:\n{}\n'.format(' '.join(docs[0])+'\n'+' '.join(docs[1])))

    """data preprocessing: word_count matrix generation"""

    counter = {}
    for doc in docs:
        for word in doc:
            counter[word] = counter.get(word, 0) + 1

    vocabulary = list(counter.keys())
    word2id = {}
    for i,word in enumerate(vocabulary):
        word2id[word] = i
    vocabulary = np.array(vocabulary)
    vocab_size = len(vocabulary)

    wordcnt_mat = np.zeros([vocab_size, doc_num])
    for i, doc in enumerate(docs):
        for word in doc:
            wordcnt_mat[word2id[word], i] += 1

    print('The frequency of the whole words: \n{}'.format(counter))
    print('vocabulary size:',vocab_size)
    print('total number of documents:', doc_num)

    """save wordcnt matrix and vocabulary to outdir"""

    np.save(outdir+'toydata_mat.npy', wordcnt_mat)
    np.save(outdir+'toydata_voc.npy', vocabulary)
    print('The generated toy data is sucessfully saved in {}'.format(outdir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toy Data Generation Paramters.')
    parser.add_argument('-D',dest = 'D', type=int,
                        help='Number of Docs',default= 100)
    parser.add_argument('--outdir', dest='outdir', action='store',
                        help='Path of the Genearted Data',default='./data/')
    args = parser.parse_args()
    toydata_generator(args.D, args.outdir)
