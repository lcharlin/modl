import numpy as np 
import time

from modl.decomposition.dict_fact import DictFact


reduction = 8 
learning_rate = 1
sample_learning_rate = 0.76
n_epochs = 50
batch_size=100
code_l1_ratio = 0
comp_l1_ratio = 0
n_threads=1
verbose = 1


print('Loading Data')
data = np.random.binomial(1, 0.1, size=(1000,1500))
print(data.dtype)

dict_fact = DictFact(reduction=reduction,
                      learning_rate=learning_rate,
                      sample_learning_rate=sample_learning_rate,
                      n_epochs = n_epochs,
                      batch_size = batch_size,
                      code_l1_ratio = code_l1_ratio,
                      comp_l1_ratio = comp_l1_ratio,
                      verbose=verbose,
                      n_threads=n_threads)

s_time = time.time()
dict_fact.fit(data)
print('Fitting time: %.3f' % (time.time()-s_time))
score = dict_fact.score(data)
print('Done fitting and evaluating')
print('Theta')
print(dict_fact.components_.shape)
#print(dict_fact.components_)
print('Beta')
print(dict_fact.code_.shape)
#print(dict_fact.code_)
