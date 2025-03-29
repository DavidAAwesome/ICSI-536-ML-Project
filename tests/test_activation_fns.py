import numpy as np
from activation_fns import Softmax

## Softmax Class Tests
def test_activate():
    n = 100
    z = np.arange(n).reshape(-1, n)
    softZ = Softmax.activate(z)
    assert sum(softZ > 0) == n
    assert np.sum(softZ) == 1

def test_deriv():
   n = 100
   z = np.arange(n).reshape(-1, n) 
   dsoftZ = Softmax.deriv(z)
   assert dsoftZ.shape == (n,n)
   # TODO: Use scipy check_grad to verify deriv