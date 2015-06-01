#ifndef __MSECTOR_H

#define __MSECTOR_H

/*minimum distance of temporary navigation point with the border of the sector (meters)*/
/*DEPRECATED*/
#define DT 3

/*Number of temporary point*/
#define NTMP 100000

/*Minimum value of F_LVL for nvp inside the sector (feets)*/
#define F_LVL_MIN 240.

/*Minimum distance between the first or the last point of the trajectory and the sector (meters)*/
/*DEPRECATED*/
#define D_AIR 30000.

/*DEPRECATED*/
#define V_THR 500.

/*If is defined read the temporary point from a file*/ /*Deprecated*/
//#define TMP_FROM_FILE

/*If It's defined cheak for intersection between flight tmp_nvp and boundary (slow)*/
//#define BOUND_CONTROL

#define CAPACITY

/*Workaround label of sector: add +1 to Labelling*/
#define WA_SECT_LABEL 1

/*Structure of shock*/
typedef struct {
	int Nshock;
	long double **shock;	
} SHOCK_t;


/*generate temporary nvp and store that in the CONF structure*/
int generate_temporary_point(CONF_t *);

/*cheak if trajectories have navigation point on the boundary of the sector
 this is necessary for our implementation of the ATC behaviour */
int cheak_traj_intersect_bound(Aircraft_t*,int,CONF_t);

/* Assigne the valure to bound[2] with the index of the nvp on the boundary */
int init_traj_intersect_bound(Aircraft_t**,int,CONF_t);

/* remove flight with more than an intersection and add nvp on the boundary */
int modify_traj_intersect_bound(Aircraft_t**,int*,CONF_t);

/* Add a nvp before st_indx and increment st_indx, 
 add also an element to velocity */
int add_nvp_st_pt(Aircraft_t *);

/*Set the flag bound[2] with the index of nvp on bound 
 and the fouth element of nvp with the flag 1 if the point is inside the polygon*/
int set_boundary_flag_onFlight(Aircraft_t **,int *,CONF_t );

/*Set base configuration for sector and trajectory*/
int init_Sector(Aircraft_t **,int *,CONF_t	*,SHOCK_t *, char *, char *);

/*cheak if a point is on the bound*/
int is_on_bound(long double *,long double **,int );

int remove_aircraft(Aircraft_t **, int *, int);

int alloc_shock(CONF_t,SHOCK_t *);

/*project the coordinate accordin to a function defined insiede*/
int project(long double *,long double *);


#endif
