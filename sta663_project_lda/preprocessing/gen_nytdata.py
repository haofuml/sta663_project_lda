import scipy.io
import numpy as np
import argparse

def nytdata_generator(outdir='./data/'):
	mat = scipy.io.loadmat('./data/nyt_data.mat')

	word_id = np.array([i[0].ravel()-1 for i in mat['Xid'].ravel()]) # matlab index starting from 1
	word_cnt = np.array([i[0].ravel() for i in mat['Xcnt'].ravel()])
	vocabulary = np.array([i[0][0] for i in mat['nyt_vocab']])

	D = len(word_cnt)
	vocab_size = len(vocabulary)
	print('total number of documents:',D)
	print('vocabulary size:',vocab_size)

	# generate word count matrix of corpus, doc_cnt.shape = vocab_size, D
	wordcnt_mat = np.zeros((vocab_size,D))
	for d in range(D):
		wordcnt_mat[word_id[d],d] = word_cnt[d]

	np.save(outdir+'nytdata_mat.npy', wordcnt_mat)
	np.save(outdir+'nytdata_voc.npy', vocabulary)
	print('The generated nyt data is sucessfully saved in {}'.format(outdir))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='NYT Data Generation Paramters.')
	parser.add_argument('--outdir', dest='outdir', action='store', 
						help='Path of the Genearted Data',default='./data/')
	args = parser.parse_args()
	nytdata_generator(args.outdir)

