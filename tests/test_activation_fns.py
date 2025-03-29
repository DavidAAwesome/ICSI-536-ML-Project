import numpy as np
from activation_fns import Softmax
from scipy.optimize import check_grad

## Softmax Class Tests
def test_activate():
    n = 100
    z = np.arange(n).reshape(1, n)
    softZ = Softmax.activate(z)
    assert np.all(softZ > 0)
    assert np.sum(softZ) == 1

def test_deriv():
   n = 4
   np.random.seed(42)
   z = np.random.rand(n)
   error = check_grad(Softmax.activate, Softmax.deriv, x0=z)
   assert error < 0.001