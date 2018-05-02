# sta663_project_lda
LDA with Collapse Gibbs Sampling and Stochastic Variational Inference

[Report](https://www.overleaf.com/15867250wrvzjcyhcwxj#/60454983/)

[Project Requirement](https://github.com/cliburn/sta-663-2018/blob/master/project/FinalProject.ipynb)

## Development Environment
* Language: Python3
* Prerequisite libraries: [Scipy](http://scipy.org), [Numpy](http://numpy.org), [Jupyter Notebook](http://jupyter.org/), [Cython](http://cython.org/)

## Environment Setup
* Fetch git repo:
```shell
git clone https://github.com/haofuml/sta663_project_lda.git
cd sta663_project_lda
```
* Install packages:
```shell
pip install --index-url https://test.pypi.org/simple/ sta663_project_lda
```

## Data Preparation
* generate toy dataset:
```shell
python -m sta663_project_lda.preprocessing.gen_toydata
```
* prepare NYT dataset:
```shell
python -m sta663_project_lda.preprocessing.gen_nytdata
```

## Experiments
* Toy dataset results:
``` shell
python -m sta663_project_lda.algorithms.lda_gibbs
python -m sta663_project_lda.algorithms.lda_svi
``` 
  alternatively:

Exceute [lda_test.ipynb](https://github.com/haofuml/sta663_project_lda/blob/master/lda_test.ipynb) in jupyter notebook

* Computational efficiency comparison:

Exceute [lda_time.ipynb](https://github.com/haofuml/sta663_project_lda/blob/master/lda_time.ipynb) in jupyter notebook

* New York Times dataset results:

Exceute [lda_nytime.ipynb](https://github.com/haofuml/sta663_project_lda/blob/master/lda_nytime.ipynb) in jupyter notebook

## Results
These are the ten-top word in each topic on New York Times dataset
* collapsed gibbs method
![collapsed gibbs](https://github.com/haofuml/sta663_project_lda/blob/master/nyt_cgibbs_result.png)
* stochastic variational method
![svi](https://github.com/haofuml/sta663_project_lda/blob/master/nyt_svi_result.png)


## Reference
* [Parameter estimation for text analysis](http://www.arbylon.net/publications/text-est.pdf)
* [Latent Dirichlet Allocation](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf)
* [Collapsed Gibbs](http://www.ics.uci.edu/~newman/pubs/fastlda.pdf)
* [Stochastic Variational Inference](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf)
* [Online LDA](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf)


