import numpy as np
from kimcalculator import KIMCalculator
from ase import Atoms
import pytest

energy_ref = 71.8572003737
forces_ref =np.array(
  [[-225.12155483,  0.,  0.],
   [ 225.12155483,  0.,  0.]]
)


def test_1D():

  modelname = 'ex_model_Ar_P_Morse_07C'
  calc = KIMCalculator(modelname,debug=True)

  dimer = Atoms('ArAr', positions=[(0,0,0),(1,0,0)], cell=(10,10,10), pbc=(0,0,0))
  dimer.set_calculator(calc)

  energy = dimer.get_potential_energy()
  forces = dimer.get_forces()

  tol = 1e-6
  print energy
  print forces
  assert energy == pytest.approx(energy_ref, tol)
  assert forces == pytest.approx(forces_ref, tol)

if __name__ == '__main__':
  test_1D()
