/*
 *  mQuery.h
 *  ElsaABM
 *
 *  Created by Christian Bongiorno on 06/03/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __MQUERY_H

#define __MQUERY_H

/* BUFFER USED FOR fgets - be carful with that! */
#define R_BUFF 50000

#define DPOS 5

#define WORKAROUND_NIGHT

/*************************** STRUCTURE ***********************************/

/* STRUCTURE AIRCRAFT - have inside all the ABM need to control the aircraft*/
typedef struct {
	/*ID of the flight (unique int number)*/
	int ID;
	
	/*number of navigation point of the route*/
	int n_nvp;
	
	/*matrix of n_nvp element x 4 (lat,lon,f_lvl, bool ) the boolen is 1 if the nvp is inside the sector*/
	long double **nvp;
	
	/*vector of n_nvp elements of time value (sec) in witch the aircraft cross the i-nvp*/
	long double *time;
	
	/*vector of n_nvp-1 element of the velocity that aircraft have for each segment (nvp[i],nvp[i+1]) (m/s)*/
	long double *vel;
	
	/*matrix of {t_w x 4} elements (lat,lon,f_lvl,bool) with the same meaning of nvp
	 but rapresent the posistion of the aircraft in the time window each time increment*/
	long double **pos;
	
	/* boolean value that is 1 if the flight is active */
	int ready;
	
	/* Index of the nvp that the aircraft point the first istant on the beginning of time-step */
	int st_indx;
	
	/* Starting point for the new time_step */
	long double st_point[DPOS];
	
	/* Index of the nvp on the boundary */
	int bound[2];
	
	/*time used to cross the sector [0]: m1 [1]:m3*/ 
	long double delta_t[2];
	
}  Aircraft_t ;


/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* STRUCTURE CONF - have inside all parameteres that characterize a simulation */
typedef struct {
	//Boundary of the sector
	int Nbound;
	long double **bound;
	
	//Max angle of deviation from original trajectory
	long double max_ang;
	
	//Number of simulation
	int nsim;
	
	//temporary point used to reroute flights
	long double **tmp_nvp;
	int n_tmp_nvp;
	
	//percentage of probability to have a direct
	long double direct_thr;
	
	//Maximum Amount of delay on departures in sec, percentage of flight that can have that delay 
	long double xdelay;
	long double pdelay;
	
	//boolean flag to have the laplacian variation of velocity
	int laplacian_vel;
	
	//Numeber of shock used for the simulation
	int Nm_shock;
	long double **point_shock;
	int n_point_shock;
	long double radius;
	long double f_lvl[2];
	int lifetime;
	
	//numer of increment in a time-step
	int t_w;
	
	//size fo the time-increment (sec)
	long double t_i;
	
	//fraction of t_w after witch the alghorithm is updated
	long double t_r;
	
	//Boolean 1) shortest path 0) minimum deviation (rerouting)
	int shortest_path;
	
	/*threshold value of the safety distance between aircraft */
	long double d_thr;
	
	/*Boolean 1) Peter-Gall Projection 2) Sferic Geometry*/
	int geom;

	/*Width of distribution of noise on velocity*/
	long double sig_V;

	/* Boolean. If 1, new temporary navpoints are read from the disk. Otherwise they are generated. */
	int tmp_from_file;

	/*Main directory*/
	char *main_dir;
	
	/*safe distance for neightboors flight*/
	long double d_neigh;
	
	/*Capcity vector of n_sect+1 elements; 0 element is an infinity capacity sector*/
	int *capacy;
	int n_sect;
	
} CONF_t ;

/********************************** FUNCTION *************************************/

/* Take the file M1 with 4-dim trajectories
 * want as arguments:
 * 1) the file with M1 trajectories
 * 2) the address of a pointer to an Aircraft type
 * 
 * Return the number of Flight*/
int get_M1(char *,Aircraft_t**,CONF_t *);

/* GetBoundary from boundary file */
int get_boundary( char *, CONF_t * );

/* Get configuration from a text file and store that in CONF_t variable */
int get_configuration(char *,CONF_t *);

/* Get temporary point for shocks */
int get_temp_shock(CONF_t *);

int add_nsim_output(char *,char *,int);

/*Get the capacity for each sector*/
int get_capacity(char *,CONF_t *);


#endif
