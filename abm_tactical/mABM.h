/*
 *  mABM.h
 *  ElsaABM_v1
 *
 *  Created by Christian Bongiorno on 08/03/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __MABM_H
#define __MABM_H

#define DAY 86400.
#define SAFE 100000

/* Parametri di Laplace */
#define MU 0.0071942
#define B_LP 0.034241

#define DEBUG0

//#define PLOT 

/*Structure with useful variable 
 that is better to carry*/
typedef struct {
	int *lista;
	
	/*vector of t_w element in witch is stored the minimum distance
	 between the aircraft in the time_window each time increment*/
	long double *dist;
	
	/*candidate nvp for rerouting*/
	long double **sel_nvp_index;
	int n_sel_nvp;
	
	long double *temp_angle;

	long double *dV;
	
} TOOL_f ;

/*Print the m1 input and set the M3*/
void init_output(Aircraft_t *,int,char *);

/*Copy most of parameters of An Aircraft structure
 to another one BE CAREFUL changing element of Aircraft Obj*/
int copy_flight(Aircraft_t *, int , Aircraft_t ** );

/*DeAlloc space on flight*/
int del_flight(Aircraft_t **, int ,Aircraft_t *);

/*Simulate*/
int ABM(Aircraft_t **,int , CONF_t, SHOCK_t );

/*Save M3 file*/
void save_m3(Aircraft_t *, int ,Aircraft_t *,char *);

#endif
