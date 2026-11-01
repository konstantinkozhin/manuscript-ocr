import numpy as np
from manuscript.utils.numerics import sigmoid, softmax


def test_activations_handle_extreme_logits_without_overflow():
    with np.errstate(over='raise', invalid='raise'):
        np.testing.assert_allclose(sigmoid(np.array([-1000., 0., 1000.])), [0, .5, 1])
        np.testing.assert_allclose(softmax(np.array([[1000., 1000.], [-1000., -1000.]])), .5)
        np.testing.assert_allclose(softmax(np.zeros((2, 3)), axis=0), .5)
