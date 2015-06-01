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

//#define LS 600
//#define LS 1500
#define LS 1000

/*Maximum distance for a temporary 
 point from the starting point*/
#define DTMP_P 100000

//#define DEBUG0

//define PLOT 

/*Maximum Number of trials to solve a conflict*/
#define N_TRY 50

#define SINGLE_TOUCH

/* To use strptime and other Time features*/
#define __USE_XOPEN 
#define  _GNU_SOURCE

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
	
	/*Edgelist of neightboors flight*/
	int **neigh;
	int *n_neigh;
	
	/*Vector of the workload for each sector*/
	int *workload;
	
	/*Comodo for Aircraft array*/
	Aircraft_t *F;
	
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

/*Deacllocate Position Matrix*/
int del_flight_pos(Aircraft_t **,int , CONF_t  );

/*Deallocate Config*/
int del_conf(CONF_t *);

/*Deallocate Config*/
int del_shock(SHOCK_t *,CONF_t*);


#endif
