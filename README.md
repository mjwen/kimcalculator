# OpenKIM ASE calculator

An Atomic Simulation Environemnt (ASE) calculator that works with Knowledgebase of Interatomic Models (KIM).

## Installation

1. install from source

Clone this repo

    $ git clone https://github.com/mjwen/kimcalculator.git

and then install by

    $ pip install -e ./kimcalculator

2. pip (to come)

3. conda (to come)

To check that `kimcalculator` is sussessfully installed, do

    $ cd kimcalculator/tests
    $ pytest

and you will get something like
```
...
collected 5 items

test_1D.py .
test_calculator.py .
test_forces.py .
test_get_model_species.py .
test_update_neigh.py .

==================== 5 passed in 0.26 seconds ====================
```

## Example

After installation,  the `KIMCalculator` can be imported from the `kimcalculator` module by

    from kimcalculator import KIMCalculator

The floowing code snippet (`test_argon.py`) shows how to compute the potential energy and forces of an FCC crystal with the KIM potentail `ex_model_Ar_P_Morse_07C`.

```python
from __future__ import print_function
import numpy as np
from ase.lattice.cubic import SimpleCubic
from kimcalculator import KIMCalculator

def test_calculator():

  # create atoms object
  argon = SimpleCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(2,2,2),
                      symbol='Ar', pbc=(1,1,0), latticeconstant=3.0)

  # create calculator
  modelname = 'ex_model_Ar_P_Morse_07C'
  calc = KIMCalculator(modelname)

  # attach calculator to atoms
  argon.set_calculator(calc)

  pos = argon.get_positions()
  energy = argon.get_potential_energy()
  forces = argon.get_forces()

  print('KIM model:', modelname)
  print('\ncoords:\n', pos)
  print('\nenergy:\n', energy)
  print('\nforces:\n', forces)

if __name__ == '__main__':
  test_calculator()
```

Run this code snippet by

    $ python test_argon.py

## Contact

Mingjian Wen (wenxx151@umn.edu)