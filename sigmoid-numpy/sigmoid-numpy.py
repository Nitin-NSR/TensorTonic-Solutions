import numpy as np

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
    ## This avoids computing extremely large exponentials.
## This technique is used in deep learning frameworks.