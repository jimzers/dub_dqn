import torch as T
import numpy as np

a = np.arange(10000, dtype=np.float32)
b = np.random.choice(np.arange(99), 32, replace=False)
print('BATCH')
print(b)
print('state batch')
print(T.tensor(a[b]))
