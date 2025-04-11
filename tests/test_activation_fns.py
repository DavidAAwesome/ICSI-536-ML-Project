import numpy as np
from nottensorflow.activation_fns import Softmax, ReLU, Sigmoid
from scipy.optimize import check_grad

np.random.seed(42) # Hardcode seed for reproducible results

def test_softmax_activate():
    n = 100
    z = np.random.rand(n)
    forward(z)
    assert (softZ > 0).all()
    assert np.absolute(np.sum(softZ) - 1) < 0.001

def test_softmax_deriv():
   n = 100
   z = np.random.rand(n)
   error = check_grad(Softmax.forward, Softmax.backward, x0=z)
   assert error < 0.001

def test_ReLU_activate():
    n = 100
    z = np.random.rand(n)
    reluZ = ReLU.forward(z)
    assert (reluZ >= 0).all()

def test_ReLU_deriv():
    n = 100
    z = np.random.rand(n)
    scalar_ReLU = lambda x: np.sum(ReLU.forward(x)) # `check_grad` expects scalar-valued fn.
    error = check_grad(scalar_ReLU, ReLU.backward, x0=z)
    assert error < 0.001

def test_sigmoid_activate():
    n = 100
    z = np.random.rand(n)
    sigZ = Sigmoid.forward(z)
    assert ((sigZ > 0) & (sigZ < 1)).all()

def test_sigmoid_deriv():
    n = 100
    z = np.random.rand(n)
    scalar_Sigmoid = lambda x: np.sum(Sigmoid.forward(x)) # `check_grad` expects scalar-valued fn.
    error = check_grad(scalar_Sigmoid, Sigmoid.backward, x0=z)
    assert error < 0.001