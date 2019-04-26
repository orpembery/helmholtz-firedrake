import numpy as np

m_eps = np.finfo(float).eps
alpha = np.log(m_eps/(1.0-m_eps))

for eps in [1.0,0.5,0.1,0.05]:
    print(1.0/alpha + eps/2.0 - np.sqrt(1.0 + alpha**2 * eps**2/4.0)/alpha)
