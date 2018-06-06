"""
To test that the calculator can produce correct energy and forces.
"""

import numpy as np
from kimcalculator import KIMCalculator
from ase.lattice.cubic import SimpleCubic, FaceCenteredCubic
import pytest

energy_ref = 19.7196709065
forces_ref = np.array(
  [[ -0.33209865,  -13.98929271,  -13.98929271],
   [  0.18090261,   13.9896848 ,  -13.98691618],
   [  0.18090261,  -13.98691618,   13.9896848 ],
   [ -0.02970657,   13.98652409,   13.98652409]]
)

def test_forces():

  # create calculator
  modelname = 'ex_model_Ar_P_Morse_07C'
  calc = KIMCalculator(modelname, debug=True)

  # create an FCC crystal
  argon = FaceCenteredCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(1,1,1),
                            symbol='Ar', pbc=(1,0,0), latticeconstant=3.0)

  # perturb the x coord of the first atom
  argon.positions[0,0] += 0.01

  # attach calculator to the atoms
  argon.set_calculator(calc)

  # get energy and forces
  energy = argon.get_potential_energy()
  forces = argon.get_forces()

  tol = 1e-6
  assert energy == pytest.approx(energy_ref, tol)
  assert forces == pytest.approx(forces_ref, tol)


if __name__ == '__main__':
  test_forces()
