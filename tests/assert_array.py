import numpy as np
import pytest

def assert_1d_array(A, B, tol=1e-6):
  A = np.array(A)
  B = np.array(B)
  assert A.shape[0] == B.shape[0]
  for a,b in zip(A,B):
    assert a == pytest.approx(b, tol)

def assert_2d_array(A, B, tol=1e-6):
  A = np.array(A)
  B = np.array(B)
  assert A.shape[0] == B.shape[0]
  assert A.shape[1] == B.shape[1]
  for a,b in zip(A,B):
    for x,y in zip(a,b):
      assert x == pytest.approx(y, tol)
