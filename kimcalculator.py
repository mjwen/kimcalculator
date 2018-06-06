from __future__ import absolute_import, division, print_function
import numpy as np
from ase.calculators.calculator import Calculator
import kimpy
import neighlist as nl
from species_name_map import species_name_map


__version__ = '2.0.0'
__author__ = 'Mingjian Wen'


class KIMCalculator(Calculator):
  """ An ASE calculator to work with KIM interatomic models.

  Parameter
  ---------

  modelname: str
    KIM model name

  neigh_skin_ratio: double
    The neighbor list is build using r_neigh = (1+neigh_skin_ratio)*rcut.

  padding_need_neigh: bool
    Flag to indicate whether need to create neighbors for padding atoms.

  debug: bool
    Flag to indicate whether to enable debug mode to print extra information.
  """

  # TODO we can support `potential_energies` and `stress` as well, depending on KIM model
  implemented_properties = ['energy', 'forces']


  def __init__(self, modelname, neigh_skin_ratio=0.2, padding_need_neigh=False,
               debug=False, **kwargs):
    Calculator.__init__(self, **kwargs)

    self.modelname = modelname
    self.debug = debug

    # neigh attributes
    if neigh_skin_ratio < 0:
      neigh_skin_ratio = 0
    self.neigh_skin_ratio = neigh_skin_ratio
    self.neigh = None
    self.skin = None
    self.cutoff = None
    self.last_update_positions = None

    # padding atoms related
    self.padding_need_neigh = padding_need_neigh
    self.num_contributing_particles = None
    self.num_padding_particles = None
    self.padding_image_of = None

    # model and compute arguments objects
    self.kim_model = None
    self.compute_arguments = None

    # model input
    self.num_particles = None
    self.species_code = None
    self.particle_contributing = None
    self.coords = None

    # model output
    # TODO we may want to support potential_energies and stress as well
    self.energy = None
    self.forces = None

    # initialization flags
    self.kim_initialized = False
    self.neigh_initialized = False


  def set_atoms(self, atoms):
    """Initialize KIM and neighbor list.
    This is called by set_calculator() of Atoms instance.

    Note that set_calculator() may be called multiple times by different Atoms instance.

    Parameter
    ---------

    atoms: ASE Atoms instance
    """
    self.init_kim(atoms)
    self.init_neigh(atoms)  #TODO since this can be attached to different atoms object,
                            # think about use different compute_arguments for each
                            # may not be good

  def init_kim(self, atoms):
    """Initialize KIM.

    Parameter
    ---------

    atoms: ASE Atoms instance
    """

    if self.kim_initialized:
      return

    # create model
    # TODO check what unit system ASE works with
    # TODO ask Ryan can we pass `unused`. Well we can test it
    units_accepted, kim_model, error = kimpy.model.create(
       kimpy.numbering.zeroBased,
       kimpy.length_unit.A,
       kimpy.energy_unit.eV,
       kimpy.charge_unit.e,
       kimpy.temperature_unit.K,
       kimpy.time_unit.ps,
       self.modelname
    )
    check_error(error, 'kimpy.model.create')
    if not units_accepted:
      report_error('requested units not accepted in kimpy.model.create')
    self.kim_model = kim_model


    # units
    if self.debug:
      l_unit,e_unit,c_unit,te_unit,ti_unit = kim_model.get_units()
      check_error(error, 'kim_model.get_units')
      print('Length unit is:', str(l_unit))
      print('Energy unit is:', str(e_unit))
      print('Charge unit is:', str(c_unit))
      print('Temperature unit is:', str(te_unit))
      print('Time unit is:', str(ti_unit))
      print()


    # create compute arguments
    self.compute_arguments, error = kim_model.compute_arguments_create()
    check_error(error, 'kim_model.compute_arguments_create')


    # check compute arguments
    num_compute_arguments = kimpy.compute_argument_name.get_number_of_compute_arguments()
    if self.debug:
      print('Number of compute_arguments:', num_compute_arguments)

    for i in range(num_compute_arguments):
      name,error = kimpy.compute_argument_name.get_compute_argument_name(i)
      check_error(error, 'kim_model.get_compute_argument_name')

      dtype,error = kimpy.compute_argument_name.get_compute_argument_data_type(name)
      check_error(error, 'kim_model.get_compute_argument_data_type')

      support_status, error = self.compute_arguments.get_argument_support_status(name)
      check_error(error, 'compute_argument.get_argument_support_status')

      if self.debug:
        n_space_1 = 21 - len(str(name))
        n_space_2 = 7 - len(str(dtype))
        print('Compute Argument name "{}" '.format(name) + ' '*n_space_1 +
              'is of type "{}" '.format(dtype) + ' '*n_space_2 +
              'and has support status "{}".'.format(support_status) )

      # can handle energy, force, particle energy, and virial as a required arg
      if support_status == kimpy.support_status.required:
        if (name != kimpy.compute_argument_name.partialEnergy or
            name != kimpy.compute_argument_name.partialForces or
            name != kimpy.compute_argument_name.partialParticleEnergy or
            name != kimpy.compute_argument_name.partialVirial):
          report_error('Unsupported required ComputeArgument {}'.format(name))


    #check compute callbacks
    num_callbacks = kimpy.compute_callback_name.get_number_of_compute_callbacks()
    if self.debug:
      print()
      print('Number of callbacks:', num_callbacks)

    for i in range(num_callbacks):

      name,error = kimpy.compute_callback_name.get_compute_callback_name(i)
      check_error(error, 'kim_model.get_compute_callback_name')

      support_status, error = self.compute_arguments.get_callback_support_status(name)
      check_error(error, 'compute_argument.get_callback_support_status')

      if self.debug:
        n_space = 18 - len(str(name))
        print('Compute callback "{}"'.format(name) + ' '*n_space +
              'has support status "{}".'.format(support_status))

      # cannot handle any "required" callbacks
      if support_status == kimpy.support_status.required:
        report_error('Unsupported required ComputeCallback: {}'.format(name))


    # set cutoff
    model_influence_dist = kim_model.get_influence_distance()
    self.skin = self.neigh_skin_ratio * model_influence_dist
    self.cutoff = (1+self.neigh_skin_ratio) * model_influence_dist
    # TODO we need to make changes to support multiple cutoffs
    model_cutoffs = kim_model.get_neighbor_list_cutoffs()
    if(model_cutoffs.size != 1):
      report_error('too many cutoffs')
    if self.debug:
      print()
      print('Model influence distance:', model_influence_dist)
      print('Number of cutoffs:', model_cutoffs.size)
      print('Model cutoffs:', model_cutoffs)
      print()

    self.kim_initialized = True



  def init_neigh(self, atoms):
    """Initialize neighbor list.

    Parameter
    ---------

    atoms: ASE Atoms instance
    """
    if self.neigh_initialized:
      return

    # create neigh object
    neigh = nl.initialize()
    self.neigh = neigh

    # register get neigh callback
    error = self.compute_arguments.set_callback_pointer(
        kimpy.compute_callback_name.GetNeighborList,
        kimpy.language_name.cpp,  #TODO ask Ryan to add python
        nl.get_neigh_kim(),
        neigh
      )
    check_error(error, 'kimpy.compute_argument.set_callback_pointer')

    self.neigh_initialized = True


  def update_kim(self, atoms):
    """ Register model input and output data pointers.

    Parameter
    ---------

    atoms: ASE Atoms instance
    """

    # model output
    self.energy = np.array([0.], dtype=np.double)
    self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)

    # register argument
    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.numberOfParticles, self.num_particles)
    check_error(error, 'kimpy.compute_argument.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.particleSpeciesCodes, self.species_code)
    check_error(error, 'kimpy.compute_argument.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.particleContributing, self.particle_contributing)
    check_error(error, 'kimpy.compute_argument.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.coordinates, self.coords)
    check_error(error, 'kimpy.compute_argument.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.partialEnergy, self.energy)
    check_error(error, 'kimpy.compute_argument.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.partialForces, self.forces)
    check_error(error, 'kimpy.compute_argument.set_argument_pointer')



  def update_neigh(self, atoms):
    """Create neighbor list and model input.
    Parameter
    ---------

    atoms: ASE Atoms instance
    """

    # get info from Atoms object
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    contributing_coords = atoms.get_positions()
    contributing_species = atoms.get_chemical_symbols()
    num_contributing = atoms.get_number_of_atoms()
    self.num_contributing_particles = num_contributing


    # species support and code
    unique_species = list(set(contributing_species))
    species_map = dict()
    for s in unique_species:
      species_support, code, error = self.kim_model.get_species_support_and_code(
         species_name_map[s])
      check_error(error or not species_support, 'kim_model.get_species_support_and_code')
      species_map[s] = code
      if self.debug:
        print('Species {} is supported and its code is: {}'.format(s, code))
    contributing_species_code = np.array(
        [species_map[s] for s in contributing_species], dtype=np.intc)


    if any(pbc):  # need padding atoms
      # create padding atoms
      padding_coords,padding_species_code,self.padding_image_of,error = nl.create_paddings(
          self.cutoff, cell, pbc, contributing_coords, contributing_species_code)
      check_error(error, 'nl.create_paddings')
      num_padding = padding_species_code.size

      self.num_particles = np.array([num_contributing + num_padding], dtype=np.intc)
      tmp = np.concatenate((contributing_coords, padding_coords))
      self.coords = np.asarray(tmp, dtype=np.double)
      tmp = np.concatenate((contributing_species_code, padding_species_code))
      self.species_code = np.asarray(tmp, dtype=np.intc)
      self.particle_contributing = np.ones(self.num_particles[0], dtype=np.intc)
      self.particle_contributing[num_contributing:] = 0
      need_neigh = np.ones(self.num_particles[0], dtype=np.intc)
      if not self.padding_need_neigh:
        need_neigh[num_contributing:] = 0

    else:  # do not need padding atoms
      self.padding_image_of = np.array([])
      self.num_particles = np.array([num_contributing], dtype=np.intc)
      self.coords = np.array(contributing_coords, dtype=np.double)
      self.species_code = np.array(contributing_species_code, dtype=np.intc)
      self.particle_contributing = np.ones(num_contributing, dtype=np.intc)
      need_neigh = self.particle_contributing

    if self.debug:
      # write configuratons with paddings
      write_extxyz(cell, self.species_code, self.coords, fname='config.xyz')

    # create neighborlist
    error = nl.build(self.neigh, self.cutoff, self.coords, need_neigh)
    check_error(error, 'nl.build')


  def update_kim_coords(self, atoms):
    """Update the atom positions in self.coords, which is registered in KIM.

    Parameter
    ---------

    atoms: ASE Atoms instance
    """

    # displacement of contributing atoms
    disp_contrib = atoms.positions - self.last_update_positions
    if self.padding_image_of.size != 0:
      # displacement of padding atoms
      disp_pad = disp_contrib[self.padding_image_of]
      # displacement of all atoms
      disp = np.concatenate((disp_contrib, disp_pad)).ravel().astype(np.double)
    else:
      disp = disp_contrib.ravel().astype(np.double)
    # update coords in KIM
    self.km_coords += disp


  def calculate(self, atoms=None,
                properties=['energy'],
                system_changes=['positions', 'numbers', 'cell', 'pbc']):
    """
    Inherited method from the ase Calculator class that is called by get_property().

    Parameters
    ----------

    atoms: ASE Atoms instance

    properties: list of str
      List of what needs to be calculated.  Can be any combination
      of 'energy' and 'forces'.

    system_changes: list of str
      List of what has changed since last calculation.  Can be
      any combination of these six: 'positions', 'numbers', 'cell',
      and 'pbc'.
    """

    Calculator.calculate(self, atoms, properties, system_changes)

    need_update_neigh = True
    if len(system_changes) == 1 and 'positions' in system_changes: # only pos changes
      if self.last_update_positions is not None:
        a = self.last_update_positions
        b = atoms.positions
        if a.shape == b.shape:
          delta = np.linalg.norm(a - b, axis=1)
          ind = np.argpartition(delta, -2)[-2:]   # indices of the two largest element
          if sum(delta[ind]) <= self.skin:
            need_update_neigh = False

    # update KIM API input data and neighbor list if necessary
    if system_changes:
      if need_update_neigh:
        self.update_neigh(atoms)
        self.update_kim(atoms)
        self.last_update_positions = atoms.positions
        if self.debug:
          print ('neighbor list updated')
      else:
        self.update_kim_coords(atoms)

      error = self.kim_model.compute(self.compute_arguments)
      check_error(error, 'kim_model.compute')

      ## TODO define a new binding function to deal with this
      #release_GIL = False
      #if 'GIL' in atoms.info:
      #  if atoms.info['GIL'] == 'off':
      #    release_GIL = True
      #status = km.model_compute(self.pkim, release_GIL)

    energy = self.energy[0]
    forces = self.forces
    forces = assemble_padding_forces(forces, self.num_contributing_particles,
        self.padding_image_of)

    # return values
    self.results['energy'] = energy
    self.results['forces'] = forces



  def get_kim_model_supported_species(self):
    return get_model_species_list(self.modelname)


  def __expr__(self):
    """Print this object shows the following message."""
    return 'KIMCalculator(modelname = {})'.format(self.modelname)


  def __del__(self):
    """Garbage collection for the KIM model object and neighbor list object."""

    # free neighbor list
    if self.neigh_initialized:
      nl.clean(self.neigh)

    # free compute arguments
    if self.kim_initialized:
      error = self.kim_model.compute_arguments_destroy(self.compute_arguments)
      check_error(error, 'kim_model.compute_arguments_destroy')

      # free kim model
      kimpy.model.destroy(self.kim_model)



def get_model_species_list(modelname):
  """Create a temporary KIM object to inqure the supported species of the model.

  Return
  ------

  species_list: list of str
    A list of all the supported species by the KIM model.
  """

  pkim, status = km.model_info(modelname)
  if status != km.STATUS_OK:
    #km.report_error('km.model_info', status)
    raise KIMCalculatorError('km.model_info')

  # number of model species
  numberSpecies, maxStrLen, status = km.get_num_model_species(pkim)
  if status != km.STATUS_OK:
    #km.report_error('km.get_num_model_species', status)
    raise KIMCalculatorError('km.get_num_model_species')

  # get species list
  species_list = []
  for i in xrange(numberSpecies):
    spec, status = km.get_model_species(pkim, i)
    if status != km.STATUS_OK:
      #km.report_error('km.get_model_species', status)
      raise KIMCalculatorError('km.get_model_species')
    species_list.append(spec)

  # destroy the temporary model and the KIM object
  status = km.free(pkim)
  if status != km.STATUS_OK:
    #km.report_error('km.free', status)
    raise KIMCalculatorError('km.free')

  return species_list


def assemble_padding_forces(forces, num_contributing, padding_image_of):
  """
  Assemble forces on padding atoms back to contributing atoms.

  Parameters
  ----------

  forces: 2D array
    forces on both contributing and padding atoms

  num_contributing: int
    number of contributing atoms

  padding_image_of: 1D int array
    atom number, of which the padding atom is an image


  Return
  ------
    Total forces on contributing atoms.
  """

  total_forces = forces[:num_contributing]

  has_padding = True if padding_image_of.size != 0 else False

  if has_padding:

    pad_forces = forces[num_contributing:]
    num_padding = pad_forces.shape[0]

    if num_contributing < num_padding:
      for i in xrange(num_contributing):
        # indices: the indices of padding atoms that are images of contributing atom i
        indices = np.where(padding_image_of == i)
        total_forces[i] += np.sum(pad_forces[indices], axis=0)
    else:
      for i in xrange(num_padding):
        total_forces[padding_image_of[i]] += pad_forces[i]

  return total_forces



def write_extxyz(cell, species, coords, fname='config.xyz'):
  """Output configurations as xyz file.

  Parameters
  ----------

  cell: 3x3 array
    Supercell lattice vectors.

  species: list of str
    atom species

  coords: 2D array
    atomic coordinates
  """
  with open (fname, 'w') as fout:
    # first line (num of atoms)
    natoms = len(species)
    fout.write('{}\n'.format(natoms))

    # second line
    # lattice
    fout.write('Lattice="')
    for line in cell:
      for item in line:
        fout.write('{} '.format(item))
    fout.write('" ')
    # properties
    fout.write('Properties=species:S:1:pos:R:3\n')

    # species, coords
    if natoms != len(coords):
      msg = 'Number of atoms is inconsistent from species and coords.\n'
      msg += 'len(specis) = {}\n'.format(natoms)
      msg += 'len(coords) = {}'.format(len(coords))
      report_error(msg)
    for i in range(natoms):
      fout.write('{:<d} '.format(species[i]))
      fout.write('{:12.5e} '.format(coords[i][0]))
      fout.write('{:12.5e} '.format(coords[i][1]))
      fout.write('{:12.5e}\n'.format(coords[i][2]))


class KIMCalculatorError(Exception):
  def __init__(self, msg):
    self.msg = msg
  def __str__(self):
    return self.msg

def check_error(error, msg):
  if error != 0 and error is not None:
    raise KIMCalculatorError('Calling "{}" failed.\n'.format(msg))

def report_error(msg):
  raise KIMCalculatorError(msg)

