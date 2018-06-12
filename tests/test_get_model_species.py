from __future__ import print_function
import numpy as np
from kimcalculator import KIMCalculator
import kimpy
from ase.lattice.cubic import SimpleCubic, FaceCenteredCubic
import pytest


def test_main():

  # create calculator
  modelname = 'ex_model_Ar_P_Morse_07C'
  #modelname = 'Three_Body_Stillinger_Weber_CdTeZnSeHgS__MO_000000111111_000'

  calc = KIMCalculator(modelname)
  species = calc.get_kim_model_supported_species()

  print ('The species supported by model "{}" is (are): {}'.format(modelname, species))
  assert species == ['Ar']

if __name__ == '__main__':
  test_main()
