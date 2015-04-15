/*
 *  mUtility.h
 *  ElsaABM
 *
 *  Created by Christian Bongiorno on 06/03/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __MUTILITY_H

#define __MUTILITY_H

#define RH 6371000.
//#define RH 100.

#define PI 3.14159265358979323846264338327950288419716939937510582097

#define SGL 0.0000001
#define SGL2 0.000000001

//#define EUCLIDEAN


/*alloc a long double array of long double*/
long double **falloc_matrix(int , int );

/*Alloc an array of long double*/
long double *falloc_vec(int );

/*Alloc an int matrix */
int **ialloc_matrix(int , int );

/*Alloc an array of int */
int *ialloc_vec(int );

/*print on screen the message and kill the program*/
void BuG(char*);

/*Calculate the aversine distance in meters for two point in lat,lon*/
long double haversine_distance(long double *, long double *);

/*Transform deg in radian*/
long double rad(long double);

/*Sort a Matrix respect to the element on the First column*/
void q_sort(long double **, int , int );

/* check if two segment intersects */
int segments_intersect(long double *,long double *, long double *, long double *);

/* check if a point is inside a Polygon */
int point_in_polygon(long double *,long double **,int );

/* extract a random long double value between minimum and maximum */
long double frand(long double ,long double );

/* Extract an int number between max and 0 */
int	irand(int );

/*cheak se il punto e compreso nel segmento*/
int isbetween(long double *,long double *, long double *);

/*Find the intersection between rect passing cross 4 point on the sphere*/
int find_intersection(long double *,long double *,long double *,long double *,long double *);

/*Free bi-dimensional matrix*/
void ffree_2D(long double **,int );

/*Free bi-dimensional int matrix*/
void ifree_2D(int **,int );


/*Free Tri-dimensional matrix*/
void ffree_3D(long double ***,int ,int );

/*Find the st_indx for a point in the sequenze of nvp */
int find_st_indx(long double **,int ,long double *,int,long double *);

/*Mix a list of int element*/
void mischia(int *,int );

/*Calculate coordinates*/
int coord3(long double , long double , long double *, long double ,long double *);

int coord(long double ,long double *,long double *,long double *,long double );

/*Calculate angle between direction*/
long double angle_direction(long double *,long double *,long double *);

/*Distance Point-segment on the sphere*/
long double distance_point_segment(long double *,long double *,long double *);

/*Find the intersection between a greatcircle passing trow two points and calculate the distance*/
int proj_point_to_circle(long double *,long double *,long double *,long double *,long double *);

/*Transform radiant in degree*/
long double deg(long double );

long double mean(long double *,int );

/*Transform time in seconds in a vector H,M,S,micrS*/
int time_to_int(long double ,int *);

/*Find the next index respect a point on a rote*/
int find_p_indx(long double **,int ,long double *);

/*Evaluate the euclidean distanca in 2Dim*/
long double euclid_dist2d(long double *,long double *);


/*cheak se il punto e compreso nel segmento EUCLIDEAN*/
int eucl_isbetween(long double *,long double *, long double *);

long double eucl_angle_direction(long double *,long double *, long double *);

#endif
