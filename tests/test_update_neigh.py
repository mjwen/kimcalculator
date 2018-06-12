"""Test update neigh is called appropriately.

For the first argon atoms, step 0 and step 2 will call update_neigh, but step 1
and step 3 will call update_kim_coords.

For the second argon atoms, all steps will call update_neigh.

Recall that neigh is updated when the sum of the two largest displacements is
larger than skin.
"""

from __future__ import print_function, division
import numpy as np
from kimcalculator import KIMCalculator
from ase.lattice.cubic import SimpleCubic, FaceCenteredCubic
import pytest


def test_main():

  # create calculator
  #modelname = 'ex_model_Ar_P_MLJ_C'
  modelname = 'ex_model_Ar_P_Morse_07C'

  calc = KIMCalculator(modelname, debug=True)

  # create an FCC crystal
  argon = FaceCenteredCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(1,1,1),
                             symbol='Ar', pbc=(1,0,0), latticeconstant=3.0)

  # perturb the x coords of the first atoms
  argon.positions[0,0] += 0.01

  # attach the calculator to the atoms object
  argon.set_calculator(calc)

  for i in range(4):

    print ('step', i)
    # get energy and forces
    energy = argon.get_potential_energy()
    forces = argon.get_forces()

    # rigidly move the atoms
    argon.positions[:,0] += 1.629/2.   # the cutoff skin is 1.63


  # create an FCC crystal with no periodic BC
  argon = FaceCenteredCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(2,1,1),
                             symbol='Ar', pbc=(0,0,0), latticeconstant=3.0)

  # attach the SAME calculator to the new atoms object
  argon.set_calculator(calc)


  for i in range(4):

    print('step', i)
    # get energy and forces
    energy = argon.get_potential_energy()
    forces = argon.get_forces()

    # rigidly move the atoms
    argon.positions[:,0] += 1.631/2.   # the cutoff skin is 1.63


if __name__ == '__main__':
  test_main()
