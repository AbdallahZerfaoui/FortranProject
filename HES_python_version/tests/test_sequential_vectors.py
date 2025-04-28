import pytest
import numpy as np
from src.vectors import SequentialVector

def test_dot():
    v1 = SequentialVector(10)
    v2 = SequentialVector(10)
    v_too_long = SequentialVector(11)
    v_too_short = SequentialVector(9)
    v1.data = np.arange(1, 11)
    v2.data = np.arange(10, 0, -1)
    
    # 1. Test dot product with same size vectors
    assert v1.dot(v2) == np.dot(v1.data, v2.data)
    assert isinstance(v1.dot(v2), float)


    # 2. Test dot product with different size vectors
    with pytest.raises(ValueError) as excinfo_long:
        v1.dot(v_too_long)
    assert "Vectors must be of the same size" in str(excinfo_long.value)

    with pytest.raises(ValueError) as excinfo_short:
        v1.dot(v_too_short)
    assert "Vectors must be of the same size" in str(excinfo_short.value)

def test_axpy():
    v1 = SequentialVector(10)
    v2 = SequentialVector(10)
    v_too_long = SequentialVector(11)
    v_too_short = SequentialVector(9)
    v1.data = np.arange(1, 11)
    v2.data = np.arange(10, 0, -1)

    # 1. Test axpy with same size vectors
    alpha = 2.0
    expected_result = v1.data + alpha * v2.data
    v1.axpy(alpha, v2)
    assert np.array_equal(v1.data, expected_result)

    # 2. Test axpy with different size vectors
    with pytest.raises(ValueError) as excinfo_long:
        v1.axpy(alpha, v_too_long)
    assert "Vectors must be of the same size" in str(excinfo_long.value)

    with pytest.raises(ValueError) as excinfo_short:
        v1.axpy(alpha, v_too_short)
    assert "Vectors must be of the same size" in str(excinfo_short.value)

def test_scale():
    v = SequentialVector(10)
    v.data = np.arange(1, 11)

    # 1. Test scale operation
    alpha = 0.5
    expected_result = v.data * alpha
    v.scale(alpha)
    assert np.array_equal(v.data, expected_result)

    # 2. Test scale with zero
    alpha = 0.0
    expected_result = np.zeros(10)
    v.scale(alpha)
    assert np.array_equal(v.data, expected_result)

def test_norm():
    v = SequentialVector(10)
    v.data = np.array([3, 4, 0, 0, 0, 0, 0, 0, 0, 0])

    # 1. Test norm calculation
    expected_norm = np.linalg.norm(v.data)
    assert v.norm() == expected_norm

    # 2. Test norm of zero vector
    v.data = np.zeros(10)
    assert v.norm() == 0.0

    # 3. Test norm of negative values
    v.data = np.array([-3, -4, -5, -6, -7, -8, -9, -10])
    expected_norm = np.linalg.norm(v.data)
    assert v.norm() == expected_norm