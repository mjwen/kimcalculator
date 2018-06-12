"""
Test the calculator can be attached to multiple Atoms object.
"""

from __future__ import print_function
import numpy as np
from kimcalculator import KIMCalculator
from ase.lattice.cubic import SimpleCubic, FaceCenteredCubic
import time


def print_values(atoms, msg):
  # compute energy and forces
  start = time.time()
  pos = atoms.get_positions()
  energy = atoms.get_potential_energy()
  forces = atoms.get_forces()
  print()
  print('='*80)
  print(msg)
  print('\ncoords:\n', pos)
  print('\nenergy:\n', energy)
  print('\nforces:\n', forces)
  print ('\nrunning time:', time.time() - start)


def test_calculator():

  # create calculator
  #modelname = 'ex_model_Ar_P_MLJ_C'
  modelname = 'ex_model_Ar_P_Morse_07C'

  calc = KIMCalculator(modelname)

  # create a SC cyrstal
  argon = SimpleCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(2,2,2),
                      symbol='Ar', pbc=(1,1,0), latticeconstant=3.0)

  # attach calculator to atoms
  argon.set_calculator(calc)

  # compute energy and forces
  print_values(argon, 'SC argon, pbc=(1,1,0)')

  # change pbc, and then compute energy and forces
  argon.set_pbc([0,0,0])
  print_values(argon, 'SC argon, pbc=(0,0,0)')

  # create a FCC crystal
  argon2 = FaceCenteredCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(2,2,2),
                             symbol='Ar', pbc=(1,1,0), latticeconstant=3.0)

  # attach the SAME calculator to the new atoms
  argon2.set_calculator(calc)

  # compute energy and forces
  print_values(argon2, 'FCC argon, pbc=(1,1,0)')



if __name__ == '__main__':
  test_calculator()
