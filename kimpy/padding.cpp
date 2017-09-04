#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "padding.h"

#define DIM 3

/*
  creating padding atoms according to PBC and cutoff.
*/

/* norm of a 3-element vector */
double norm(double* a) {
  return sqrt(a[0]*a[0]+ a[1]*a[1]+ a[2]*a[2]);
}

/* dot product of two 3-element vectors */
double dot(double *a, double *b)
{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/* cross product of two 3-element vectors */
void cross(double *a, double *b, double* axb)
{
  axb[0] = a[1]*b[2] - a[2]*b[1];
  axb[1] = a[2]*b[0] - a[0]*b[2];
  axb[2] = a[0]*b[1] - a[1]*b[0];
}

/* determinant of a 3 by 3 matrix */
double det(double *mat)
{
  return (mat[0]*mat[4]*mat[8] - mat[0]*mat[5]*mat[7]
      - mat[1]*mat[3]*mat[8] + mat[1]*mat[5]*mat[6]
      + mat[2]*mat[3]*mat[7] - mat[2]*mat[4]*mat[6]);
}


double det2(double a11, double a12, double a21, double a22)
{
  return (a11*a22) - (a12*a21);
}

/*inverse of a 3 by 3 matrix */
void inverse(double *mat, double *inv)
{
  int i;
  double dd;

  inv[0] = det2(mat[4], mat[5], mat[7], mat[8]);
  inv[1] = det2(mat[2], mat[1], mat[8], mat[7]);
  inv[2] = det2(mat[1], mat[2], mat[4], mat[5]);
  inv[3] = det2(mat[5], mat[3], mat[8], mat[6]);
  inv[4] = det2(mat[0], mat[3], mat[6], mat[8]);
  inv[5] = det2(mat[2], mat[0], mat[5], mat[3]);
  inv[6] = det2(mat[3], mat[4], mat[6], mat[7]);
  inv[7] = det2(mat[1], mat[0], mat[7], mat[6]);
  inv[8] = det2(mat[0], mat[1], mat[3], mat[4]);

  dd = det(mat);
  for (i=0; i<9; i++) {
    inv[i] /= dd;
  }
}

/* transpose of a 3 by 3 matrix */
void transpose(double *mat, double* trans)
{
  int i,j;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      trans[3*i+j] = mat[3*j+i];
    }
  }
}


/* create padding atoms */
void set_padding(int Natoms, double* cell, int* PBC, double cutoff, double** coords,
    char (**species)[4], int** pad_image, int* Npad)
{

  int i,j,k;
  double tcell[9];
  double fcell[9];
  double atom_coords[3];
  double frac_coords[DIM*Natoms];

  double xprod[3];
  double volume;
  double dist0;  /* distance of cell surfaces that are 'perpendicular' to x-axis */
  double dist1;
  double dist2;
  double ratio0;
  double ratio1;
  double ratio2;
  int size0; /* number of bins in x direction */
  int size1;
  int size2;

  int at;
  double x,y,z;
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;
  double* pad_coords = NULL;
  char (*pad_species)[4] = NULL;
  *pad_image = NULL;



  /* transform coords into fractional coords */
  transpose(cell, tcell);
  inverse(tcell, fcell);
  xmin=0;
  ymin=0;
  zmin=0;
  xmax=0;
  ymax=0;
  zmax=0;
  for (i=0; i<Natoms; i++) {
    atom_coords[0] = (*coords)[DIM*i+0];
    atom_coords[1] = (*coords)[DIM*i+1];
    atom_coords[2] = (*coords)[DIM*i+2];
    x = dot(&fcell[0],atom_coords);
    y = dot(&fcell[3],atom_coords);
    z = dot(&fcell[6],atom_coords);
    frac_coords[DIM*i+0] = x;
    frac_coords[DIM*i+1] = y;
    frac_coords[DIM*i+2] = z;
    if (x < xmin) xmin = x;
    if (y < ymin) ymin = y;
    if (z < zmin) zmin = z;
    if (x > xmax) xmax = x;
    if (y > ymax) ymax = y;
    if (z > zmax) zmax = z;
  }


  /* volume of cell */
  cross(&cell[3], &cell[6], xprod);
  volume = dot(&cell[0], xprod);

  /* distance between parallelpiped faces */
  cross(&cell[3], &cell[6], xprod);
  dist0 = volume/norm(xprod);
  cross(&cell[6], &cell[0], xprod);
  dist1 = volume/norm(xprod);
  cross(&cell[0], &cell[3], xprod);
  dist2 = volume/norm(xprod);

  ratio0 = cutoff/dist0;
  ratio1 = cutoff/dist1;
  ratio2 = cutoff/dist2;
  size0 = (int)ceil(ratio0);
  size1 = (int)ceil(ratio1);
  size2 = (int)ceil(ratio2);

  /* creating padding atoms */
  *Npad = 0;
  for (i=-size0; i<=size0; i++)
  for (j=-size1; j<=size1; j++)
  for (k=-size2; k<=size2; k++) {

    /* skip contributing atoms that we already have */
    if (i==0 && j==0 && k==0) continue;

    /* apply BC */
    if (PBC[0]==0 && i != 0) continue;
    if (PBC[1]==0 && j != 0) continue;
    if (PBC[2]==0 && k != 0) continue;

    for (at=0; at<Natoms; at++) {
      x = frac_coords[DIM*at+0];
      y = frac_coords[DIM*at+1];
      z = frac_coords[DIM*at+2];

      /* select the necessary atoms to repeate for the most outside bins */
      /* the follwing few lines can be easily understood when assuming size=1 */
      if (i == -size0 && x - xmin < (double)size0 - ratio0) continue;
      if (i == size0  && xmax - x < (double)size0 - ratio0) continue;
      if (j == -size1 && y - ymin < (double)size1 - ratio1) continue;
      if (j == size1  && ymax - y < (double)size1 - ratio1) continue;
      if (k == -size2 && z - zmin < (double)size2 - ratio2) continue;
      if (k == size2 &&  zmax - z < (double)size2 - ratio2) continue;


      /* add a padding atom */
      *Npad += 1;
      pad_coords = (double*) realloc(pad_coords, (*Npad)*DIM*sizeof(double));
      pad_species =  (char(*)[4]) realloc(pad_species, (*Npad)*sizeof(char[4]));
      *pad_image =  (int*) realloc(*pad_image, (*Npad)*sizeof(int));

      /* fractional coords of padding atom at */
      atom_coords[0] = i+x;
      atom_coords[1] = j+y;
      atom_coords[2] = k+z;

      /* absolute coords of padding atoms */
      pad_coords[DIM*(*Npad-1)+0] = dot(&tcell[0], atom_coords);
      pad_coords[DIM*(*Npad-1)+1] = dot(&tcell[3], atom_coords);
      pad_coords[DIM*(*Npad-1)+2] = dot(&tcell[6], atom_coords);

      strcpy(pad_species[*Npad-1], (*species)[at]);
      (*pad_image)[*Npad-1] = at;
    }
  }

  /* concatenate coords and species of padding to that of contributing atoms */
  *coords = (double*) realloc(*coords, DIM*(Natoms + *Npad)*sizeof(double));
  memcpy(*coords+DIM*Natoms, pad_coords, DIM*(*Npad)*sizeof(double));
  *species = (char(*)[4]) realloc((*species), (Natoms + *Npad)*sizeof(char[4]));
  memcpy(*species+Natoms, pad_species, (*Npad)*sizeof(char[4]));


  /* free local memory */
  free(pad_coords);
  free(pad_species);
}