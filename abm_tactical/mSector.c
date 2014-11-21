/*
 *  mSector.c
 *  ElsaABM_v1
 *
 *  Created by Christian Bongiorno on 07/03/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include "mQuery.h"
#include "mUtility.h"
#include "mSector.h"
#include "mTest.h"

#include<stdlib.h>
#include<stdio.h>
#include<math.h>

//Find extream value of a matrix respect to the column c. if Max=1 find max if max!=1 find min
long double _find_extrem(long double **p,int np,int c,int Max){
	int i;
	long double x;
	if(Max==1){
		for(i=1,x=p[0][c];i<np;i++) if(p[i][c]>x) x=p[i][c];
		return x;
	}
	else{
		for(i=1,x=p[0][c];i<np;i++) if(p[i][c]<x) x=p[i][c];
		return x;		
	}
}

int _point_proj(long double *P,long double *A,long double *B,long double *H){
	
	if(B[0]==A[0]){
		H[0]=B[0];
		H[1]=P[1];
		return 1;
	}
	
	long double m=(B[1]-A[1])/(B[0]-A[0]);
	long double c=A[1]-A[0]*m;
	
	H[0]=(P[0]/m + P[1] - c )/(m + 1./m);
	H[1]=m*((P[0]/m + P[1] - c )/(m + 1./m)) +c;
	
	return 1;
}

int _check_tmp_point(long double *p,CONF_t conf){
	int i;
	if(!point_in_polygon(p, conf.bound, conf.Nbound)) return 1;
	for(i=0;i<(conf.Nbound-1);i++) if( distance_point_segment(conf.bound[i],conf.bound[i+1],p)<DT ) return 1;
	
	return 0;
}

int generate_temporary_point(CONF_t *config){
	
	(*config).n_tmp_nvp=NTMP;
	
	(*config).tmp_nvp = falloc_matrix((*config).n_tmp_nvp, 2);
	int n;
#ifdef TMP_FROM_FILE
	printf("Attention! read temporary nvp from file\n");
	int i;
	char c[500];

	char* rep_tail = "/abm_tactical/config/temp_nvp.dat";
	char * rep = malloc(snprintf(NULL, 0, "%s%s", (*config).main_dir, rep_tail) + 1);
	sprintf(rep, "%s%s", (*config).main_dir, rep_tail);

	FILE *rstream=fopen(rep,"r");
	if(rstream==NULL) BuG("Impossible to open temp_nvp.dat\n");
	for(n=0;fgets(c,500,rstream);n++){
		(*config).tmp_nvp[n][0]=atof(c);
		for(i=0;c[i]!='\t';i++);
		(*config).tmp_nvp[n][1]=atof(&c[++i]);
	}
	fclose(rstream);
	free(rep);
	return 1;
#endif
	
	long double Mx= _find_extrem((*config).bound,(*config).Nbound,0,1);
	long double mx= _find_extrem((*config).bound,(*config).Nbound,0,0);
	long double My= _find_extrem((*config).bound,(*config).Nbound,1,1);
	long double my= _find_extrem((*config).bound,(*config).Nbound,1,0);
	
	for (n=0; n<(*config).n_tmp_nvp ; n++ ) {
		do{
			(*config).tmp_nvp[n][0]=frand(mx, Mx);
			(*config).tmp_nvp[n][1]=frand(my, My);
		}while ( _check_tmp_point( (*config).tmp_nvp[n],(*config) ));
	}
	FILE *wstream=fopen(rep,"w");
	for(n=0;n<(*config).n_tmp_nvp;n++) fprintf(wstream,"%Lf\t%Lf\n",(*config).tmp_nvp[n][0],(*config).tmp_nvp[n][1]);
	fclose(wstream);

	free(rep);
	
	return 1;
}

int cheak_traj_intersect_bound(Aircraft_t *flight,int Nflight,CONF_t config){
	int i,j,k,n_inter;
	for(i=0;i<Nflight;i++) {
		for(j=0,n_inter=0;j<flight[i].n_nvp;j++) 
			for(k=0;k<(config.Nbound-1);k++) 
				if(isbetween(config.bound[k], config.bound[k+1], flight[i].nvp[j])){
					n_inter++;
					break;
				}
		if(n_inter<2) BuG("The current version of the ABM require that the trajectories of all flight have two nvp on the boundary of the sector\n");
		
		if(n_inter>2) BuG("There are trajectories with more than two intersections with the boundary of the sector\n");	
	} 
	return 1;
}

int init_traj_intersect_bound(Aircraft_t **flight,int Nfligth,CONF_t config){
	int i,j,k,c;
	
	for(i=0;i<Nfligth;i++) 
		for(j=0,c=0;j<(*flight)[i].n_nvp&&c<2;j++) 
			for(k=0;k<(config.Nbound-1);k++) 
				if(isbetween(config.bound[k],config.bound[k+1],(*flight)[i].nvp[j])){
					(*flight)[i].bound[c++]=j;
					break;
				}
	return 1;
}

int remove_aircraft(Aircraft_t **fligth, int *Nfligth, int sel){
	Aircraft_t *New_fligth=(Aircraft_t*) malloc((*Nfligth-1)*sizeof(Aircraft_t));
	if(New_fligth==NULL) BuG("Memory BUG\n");
	int i,h;
	for(i=0,h=0;i<*Nfligth;i++) if(i!=sel) New_fligth[h++]=(*fligth)[i];
	free((*fligth) );
	
	(*fligth)=New_fligth;
	*Nfligth= *Nfligth -1;
	
	return 1;
}

int add_nvp(Aircraft_t *f,int *st_indx,long double *p){
	
	long double **nvp=(long double**) malloc( ((*f).n_nvp+1)*sizeof(long double*) );
	long double *vel=falloc_vec((*f).n_nvp);
	//long double *time = falloc_vec((*f).n_nvp+1 );
	
	int i;
	for(i=0;i<*st_indx;i++) {
		if(i>((*f).n_nvp-2)) BuG("Excess\n");
		nvp[i]=(*f).nvp[i];
		vel[i]=(*f).vel[i];
		//time[i]=(*f).time[i];
	}
	
	if((i-1)>((*f).n_nvp-2)) BuG("Excess\n");
	nvp[i]=p;
	nvp[i][2]=nvp[i-1][2];
	nvp[i][3]=nvp[i-1][3];

	vel[i]=(*f).vel[i-1];
	
	//time[i]= haversine_distance( nvp[i-1], nvp[i] )/ vel[i];
	
	for(;i<((*f).n_nvp-1);i++){
		if(i>((*f).n_nvp-2)) BuG("Excess\n");
		nvp[i+1]=(*f).nvp[i];
		vel[i+1]=(*f).vel[i];
		//time[i+1]=(*f).time[i];
	}
	if(i>((*f).n_nvp-1)) BuG("Excess\n");
	nvp[i+1]=(*f).nvp[i];
	//time[i+1]=(*f).time[i];
	
	free((*f).nvp);
	free((*f).vel);
	//free((*f).time);
	
	(*f).nvp=nvp;
	(*f).vel=vel;
	//(*f).time=time;
	
	(*f).n_nvp = (*f).n_nvp+1;
	(*st_indx)++;
	
	return 1;
}

int _add_nvp_bound(Aircraft_t **f,int i,int j,long double **bound,int k){
	int st_indx=j+1;
	long double *p=falloc_vec(4);

	find_intersection((*f)[i].nvp[j],(*f)[i].nvp[j+1],bound[k],bound[k+1],p);
	add_nvp( &( (*f)[i] ),&st_indx,p);
	return 1;
}

int _is_to_add(Aircraft_t f,int xp,long double **bound,int k){
	
	if( ( segments_intersect(f.nvp[xp], f.nvp[xp+1], bound[k], bound[k+1]) && !isbetween(bound[k], bound[k+1],f.nvp[xp+1]) ) && !isbetween( bound[k], bound[k+1],f.nvp[xp]) ){
		return 1;
	   }
	return 0;
} 

int modify_traj_intersect_bound(Aircraft_t **flight,int *Nflight,CONF_t config){
	int i,j;
		
	/*Change exagerate velocity: flight too much speed*/
	/*
	for(i=0;i<*Nflight;i++){
		for(j=0;j<((*flight)[i].n_nvp-1);j++) if((*flight)[i].vel[j]>V_THR) {
			(*flight)[i].vel[j]=240.;
		}
	}*/
	
	/*Add flag to nvp*/
	for(i=0;i<*Nflight;i++){
		(*flight)[i].nvp[0][3]=0;
		for(j=1;j<((*flight)[i].n_nvp-2);j++) (*flight)[i].nvp[j][3]=1;
		(*flight)[i].nvp[j][3]=0;
	}
	
	return 1;
}

int set_boundary_flag_onFlight(Aircraft_t **f,int *Nflight, CONF_t c){
	int i,j,N;
	for(i=0;i<*Nflight;i++) {
		(*f)[i].st_indx=1;
		(*f)[i].ready=0;
		for(j=0,N=0;j<((*f)[i].n_nvp-1);j++) {
			if(( point_in_polygon((*f)[i].nvp[j], c.bound, c.Nbound)||point_in_polygon((*f)[i].nvp[j+1], c.bound, c.Nbound) )&&(*f)[i].nvp[j][2]>=F_LVL_MIN){
				(*f)[i].nvp[j][3]=1.;
				}
			 else  (*f)[i].nvp[j][3]=0.;
			}
			   
//		if(point_in_polygon((*f)[i].nvp[j], c.bound, c.Nbound)) (*f)[i].nvp[j][3]=1.;
//		else  (*f)[i].nvp[j][3]=0.;
		
		for(j=0;j<((*f)[i].n_nvp-1);j++) if(point_in_polygon((*f)[i].nvp[j], c.bound, c.Nbound)||point_in_polygon((*f)[i].nvp[j+1], c.bound, c.Nbound)){
			(*f)[i].bound[0]=j;
			break;
		}
		if(j==((*f)[i].n_nvp-1)){
			remove_aircraft(f, Nflight, i--);
			printf("Removed %d Aircraft\n",(*f)[i].ID);
			continue;
		}
		//if((*f)[i].bound[0]==0) (*f)[i].bound[0]=1;
		for(++j;j<((*f)[i].n_nvp);j++) if(!point_in_polygon((*f)[i].nvp[j], c.bound, c.Nbound)&&point_in_polygon((*f)[i].nvp[j-1], c.bound, c.Nbound)) {
			(*f)[i].bound[1]=j;
			break;			
		}
		if(j==((*f)[i].n_nvp)) (*f)[i].bound[1]=j-1;
		   
		
		if((*f)[i].bound[0]==(*f)[i].bound[1]){
			printf("SOmetingh wrong\n");
			(*f)[i].bound[1]=(*f)[i].bound[1]+1;
		}
	}
	

	return 1;
}

int is_on_bound(long double *p,long double **bound,int N){
	int i;
	for(i=0;i<(N-1);i++) if(isbetween(bound[i], bound[i+1], p)) return 1;
	return 0;
}

int _alloc_shock( CONF_t conf,SHOCK_t *shock ){
	(*shock).Nshock = (int) (conf.nsim*((conf.f_lvl[1]-conf.f_lvl[0])/10)*conf.Nm_shock);
	(*shock).shock=falloc_matrix( (*shock).Nshock , 6);
	return 1;
}

int _set_cross_timeM1(Aircraft_t **f, int N){
	int i;
	for(i=0;i<N;i++) (*f)[i].delta_t[0]=(*f)[i].time[(*f)[i].bound[1]] - (*f)[i].time[(*f)[i].bound[0]];
	return 1;
}

int _alloc_flight_pos(Aircraft_t **f,int N_f,CONF_t *conf){
	int i;
	for(i=0;i<N_f;i++) (*f)[i].pos = falloc_matrix((*conf).t_w*2, 4);
	return 1;
}

int init_Sector(Aircraft_t **flight,int *Nflight,CONF_t	*config, SHOCK_t *shock,char *input_ABM){
	//get_boundary("CONF/bound_latlon.dat", config);	
	//char *main_dir = "/home/earendil/Documents/ELSA/ABM/ABM_FINAL";
	char *main_dir = "/home/earendil/Documents/ELSA/ABM/ABM_FINAL";
	char* rep_tail2 = "/abm_tactical/config/config.cfg";
	char * rep2 = malloc(snprintf(NULL, 0, "%s%s", main_dir, rep_tail2) + 1);
	sprintf(rep2, "%s%s", main_dir, rep_tail2);

	get_configuration(rep2, config);

	char* rep_tail = "/abm_tactical/config/bound_latlon.dat";
	char * rep = malloc(snprintf(NULL, 0, "%s%s", (*config).main_dir, rep_tail) + 1);
	sprintf(rep, "%s%s", (*config).main_dir, rep_tail);
	get_boundary(rep, config);	

	free(rep);
	free(rep2);

	printf("Generate Point\n");
	generate_temporary_point(config);
	
	*Nflight=get_M1(input_ABM,flight);
	
	modify_traj_intersect_bound(flight, Nflight, *config);

	//cheak_traj_intersect_bound(*flight, *Nflight, *config);
	
	//set_boundary_flag_onFlight(flight,Nflight,*config);
	
	
	
	
	//_set_cross_timeM1(flight,*Nflight);
	_alloc_flight_pos(flight,*Nflight,config);
	
	_alloc_shock(*config,shock);
	get_temp_shock(config);
	
	
	return 1;
}





