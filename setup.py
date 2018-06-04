from setuptools import setup
from distutils.sysconfig import get_config_vars


# remove `-Wstrict-prototypes' that is for C not C++
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
  if type(value) == str and '-Wstrict-prototypes' in value:
    cfg_vars[key] = value.replace('-Wstrict-prototypes', '')


def get_version(fname='kimcalculator.py'):
  with open(fname) as fin:
    for line in fin:
      line = line.strip()
      if '__version__' in line:
        v = line.split('=')[1]
        # stripe white space, and ' or " in string
        if "'" in v:
          version = v.strip("' ")
        elif '"' in v:
          version = v.strip('" ')
        break
  return version


setup(name = 'kimcalculator',
  version = get_version(),
  install_requires = ['numpy', 'ase'],

  # metadata
  author = 'Mingjian Wen',
  author_email = 'wenxx151[at]umn.edu',
  url = 'https://github.com/mjwen/kimcalculator',
  description = 'An ASE calculator to work with KIM interatomic potentials',
  #zip_safe = False,
)

