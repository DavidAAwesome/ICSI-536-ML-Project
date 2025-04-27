import numpy as np
from nottensorflow.activation_fns import Softmax, ReLU, Sigmoid
from scipy.optimize import check_grad

np.random.seed(42) # Hardcode seed for reproducible results

def test_softmax_activate():
    n = 100
    z = np.random.rand(n)
    softmax = Softmax()
    softZ = softmax.forward(z)
    assert (softZ > 0).all()
    assert np.absolute(np.sum(softZ) - 1) < 0.001

def test_softmax_deriv():
   n = 100
   z = np.random.rand(n)
   softmax = Softmax()
   softmax.forward(z)
   output_grad = np.random.rand(n)
   grad = softmax.backward(output_grad, learning_rate=0.1)
   # for softmax, the gradient should be a matrix multiplication
   s = softmax.forward(z).reshape(-1, 1)
   J = np.diagflat(s) - np.dot(s, s.T)
   expected_grad = np.dot(J, output_grad)
   assert np.allclose(grad, expected_grad)

def test_ReLU_activate():
    n = 100
    z = np.random.rand(n)
    relu = ReLU()
    reluZ = relu.forward(z)
    assert (reluZ >= 0).all()

def test_ReLU_deriv():
    n = 100
    z = np.random.rand(n)
    relu = ReLU()
    relu.forward(z)
    output_grad = np.random.rand(n)
    grad = relu.backward(output_grad, learning_rate=0.1, SGD=False)
    assert (grad >= 0).all()
    assert np.allclose(grad, output_grad * (z > 0))

def test_sigmoid_activate():
    n = 100
    z = np.random.rand(n)
    sigmoid = Sigmoid()
    sigZ = sigmoid.forward(z)
    assert ((sigZ > 0) & (sigZ < 1)).all()

def test_sigmoid_deriv():
    n = 100
    z = np.random.rand(n)
    sigmoid = Sigmoid()
    sigmoid.forward(z)
    output_grad = np.random.rand(n)
    grad = sigmoid.backward(output_grad, learning_rate=0.1)
    sig = sigmoid.forward(z)
    expected_grad = output_grad * sig * (1 - sig)
    assert np.allclose(grad, expected_grad)