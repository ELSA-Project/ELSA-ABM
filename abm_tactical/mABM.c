/*
 *  mABM.c
 *  ElsaABM_v1
 *
 *  Created by Christian Bongiorno on 08/03/14.
 *
 */
#include "mQuery.h"
#include "mUtility.h"
#include "mSector.h"

#include "mABM.h"
#include "mTest.h"

#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<math.h>
#include<time.h>
#include<float.h>

void init_output(Aircraft_t *flight,int Nflight,char *output_ABM){
	/* Non capisco a che serve l M1
	FILE *wstream=fopen("DATA/m1_data.dat","w");
	if(wstream==NULL) BuG("Impossible to save output data\n");
	
	int i,j;
	for(i=0;i<Nflight;i++){ 
		fprintf(wstream,"%d\t%d\t",flight[i].ID,flight[i].n_nvp);
		for(j=0;j<flight[i].n_nvp;j++) 
			fprintf(wstream,"%Lf,%Lf,%Lf,%Lf\t",flight[i].nvp[j][0],flight[i].nvp[j][1],flight[i].nvp[j][2],flight[i].time[j]);
		fprintf(wstream, "\n");
	}
	fclose(wstream);
	*/
	FILE *wstream = fopen(output_ABM,"w");
	if(wstream==NULL) BuG("Impossible to save output data\n");
	fclose(wstream);
	
	/*
	wstream = fopen("DATA/rer_jump_direct.dat","w");
	if(wstream==NULL) BuG("Impossible to save output data\n");
	fclose(wstream);
	*/
	return;
}

void save_m3(Aircraft_t *flight, int Nflight,Aircraft_t *Flight,char *output_ABM){

	FILE *wstream=fopen(output_ABM,"w");
	if(wstream==NULL) BuG("Impossible to save output data\n");
	
	time_t pT;
	struct tm *pTm;
	char buff[100];
	
	fprintf(wstream,"%d\tNflight\n",Nflight);
	int i,j,h,T[DPOS];
	//for(h=0;h<Nflight;h++){
		//for(i=0;i<Nflight;i++) if(Flight[h].ID==flight[i].ID) { 
		for(i=0;i<Nflight;i++) { 
			
			fprintf(wstream,"%d\t%d\t",flight[i].ID,flight[i].n_nvp);
			for(j=0;j<flight[i].n_nvp;j++){
				
				//time_to_int(flight[i].time[j],T);
				pT = (time_t) flight[i].time[j];
				pTm = localtime(&pT);
				
				strftime(buff,100,"%Y-%m-%d %H:%M:%S",pTm);
				#ifdef CAPACITY
				fprintf(wstream,"%.10LF,%.10LF,%.0Lf,%s,%d\t",flight[i].nvp[j][0],flight[i].nvp[j][1],flight[i].nvp[j][2],buff,(int) flight[i].nvp[j][DPOS-1]);
				#else
				fprintf(wstream,"%.10Lf,%.10Lf,%.0Lf,%s\t",flight[i].nvp[j][0],flight[i].nvp[j][1],flight[i].nvp[j][2],buff);				
				#endif
			}
			fprintf(wstream, "\n");
		}
	//}
	//fprintf(wstream, "---\n");
	fclose(wstream);
	
	return;
}

int del_flight(Aircraft_t **f, int N,Aircraft_t *F){
	int i,j;
	for(i=0;i<N;i++){
		//for(j=0;j<N;j++) if((F)[j].ID==(*f)[i].ID) break;
		//ffree_2D((*f)[i].nvp, (F)[j].n_nvp);
		ffree_2D((*f)[i].nvp, (*f)[i].n_nvp);
		free( (*f)[i].vel );
		free( (*f)[i].time );
	}
	free((*f));
	
	return 1;
}

int del_flight_pos(Aircraft_t **f,int N, CONF_t conf ){
	int i;
	for(i=0;i<N;i++) ffree_2D((*f)[i].pos,2*conf.t_w);
	
	return 1;
}

int del_conf(CONF_t *conf){
	ffree_2D((*conf).bound,(*conf).Nbound);
	ffree_2D((*conf).tmp_nvp,(*conf).n_tmp_nvp);
	free((*conf).main_dir);
	return 1;
	
}
int del_shock(SHOCK_t *shock,CONF_t *conf){
	
	ffree_2D((*shock).shock,(*shock).Nshock);
	ffree_2D((*conf).point_shock,(*conf).n_point_shock);

	return 1;
}


int copy_flight(Aircraft_t *F, int N, Aircraft_t **f ){
	int i,j,k;
	
	(*f)=(Aircraft_t *) malloc(N*sizeof(Aircraft_t));
	if(*f==NULL) BuG("Not enough memory\n");
	
	for(i=0;i<N;i++){
		(*f)[i].ID = F[i].ID;
		
		(*f)[i].n_nvp = F[i].n_nvp;
		
		(*f)[i].pos = F[i].pos;
		(*f)[i].ready= F[i].ready;
		
		(*f)[i].nvp = falloc_matrix((*f)[i].n_nvp, DPOS);
		for(j=0;j<(*f)[i].n_nvp;j++) for(k=0;k<DPOS;k++) (*f)[i].nvp[j][k] = F[i].nvp[j][k];
		
		(*f)[i].time = falloc_vec((*f)[i].n_nvp);
		for(j=0;j<(*f)[i].n_nvp;j++) (*f)[i].time[j] = F[i].time[j];
		
		(*f)[i].vel = falloc_vec((*f)[i].n_nvp-1);
		for(j=0;j<((*f)[i].n_nvp-1);j++) (*f)[i].vel[j] = F[i].vel[j];
		
		(*f)[i].st_indx = F[i].st_indx;
		
		(*f)[i].bound[0] = F[i].bound[0];
		(*f)[i].bound[1] = F[i].bound[1];
		
		(*f)[i].delta_t[0] = F[i].delta_t[0];
	}
	return 1;
}

int _delay_on_m1(Aircraft_t **f,int N_f,CONF_t conf){
	int i;
	for(i=0;i<N_f;i++) if(frand(0,1)<conf.pdelay)  (*f)[i].time[0]= (*f)[i].time[0] + frand(-conf.xdelay, conf.xdelay);
	return 1;
}

int _init_tool(TOOL_f *t,int N, CONF_t conf){
	int i;
	
	(*t).lista = ialloc_vec(N);
	for(i=0;i<N;i++) (*t).lista[i]=i;
	
	
	
	(*t).dist = falloc_vec(2*conf.t_w);
	

	(*t).sel_nvp_index=falloc_matrix(conf.n_tmp_nvp,2);
	if((*t).sel_nvp_index==NULL) BuG("not enough Memory\n");

	
	(*t).temp_angle=falloc_vec(conf.n_tmp_nvp);

	(*t).dV = falloc_vec(N);
	
	(*t).neigh = ialloc_matrix(N,N);
	(*t).n_neigh = ialloc_vec(N);
	
	#ifdef CAPACITY
	(*t).workload = ialloc_vec(conf.n_sect+1);
	#endif

	(*t).F = (Aircraft_t*) malloc(N*sizeof(Aircraft_t));
	if( (*t).F==NULL) BuG("No Memory\n");
	
	return 1;
}

int _del_tool(TOOL_f *t,int N,CONF_t conf){
	free((*t).lista);
	free((*t).dist);
	ffree_2D((*t).sel_nvp_index, conf.n_tmp_nvp);
	free((*t).temp_angle);
	free((*t).dV);
	ifree_2D((*t).neigh,N);
	free((*t).n_neigh);
	

	free((*t).workload);
	free((*t).F);
	
	return 1;
}

int _create_shock(SHOCK_t *sh,CONF_t conf){
	
	int i,coin, niter= (int)((conf.end_datetime - conf.start_datetime)/ (conf.t_r*conf.t_w*conf.t_i));
	for(i=0;i<(*sh).Nshock;i++){
		coin=irand(conf.n_point_shock);
		(*sh).shock[i][0]=conf.point_shock[coin][0];
		(*sh).shock[i][1]=conf.point_shock[coin][1];
		(*sh).shock[i][2]=conf.radius;
		(*sh).shock[i][3]= ((long double) irand(niter) )*(conf.t_r*conf.t_w*conf.t_i);
		(*sh).shock[i][4]= 1+((long double) irand(conf.lifetime-1) );
		(*sh).shock[i][5]= conf.f_lvl[0] + (irand(((conf.f_lvl[1]-conf.f_lvl[0])/10) ))*10.;		
	}
	return 1;
}

/*int _nvp_to_close(long double *d,long double *t_c,long double *vel,long double **nvp, long double *l_nvp,int *j,int n_nvp,long double *time,long double t){*/
int _nvp_to_close(long double *d,long double *t_c,long double *vel,long double dV, long double **nvp, long double *l_nvp,int *j,int n_nvp,long double *time,long double t){
	if(*d>=*l_nvp){
		*t_c=*t_c - *l_nvp/(vel[*j]*(1.+dV));
		(*j)=*j+1;
		if((*j+1)> (n_nvp-1)) {
			return(0);}
		time[*j]=t- *t_c;
		*d=(vel[*j]*(1.+dV))* (*t_c);
		#ifdef EUCLIDEAN
		*l_nvp=euclid_dist2d(nvp[*j],nvp[*j+1]);
		#else
		*l_nvp=haversine_distance(nvp[*j],nvp[*j+1]);
		#endif
		
		return _nvp_to_close(d,t_c,vel,dV,nvp,l_nvp,j,n_nvp,time,t);
	}
	else return 1;
}

long double _f_dist(int t,CONF_t *conf){
	/*Function for increasing the safaty distance with time*/
	return pow(t*(*conf).t_i,1) * (*conf).noise_d_thr;
	
}

/*Calculate the Position Array in the time-step . Return Zero if the Aircraft will land in the current timestep.*/
int _position(Aircraft_t *f,long double *st_point, int t_wind, long double time_step, long double t_r, long double dV){
	int i,j;
	long double l_nvp,d,t_c;
	
	#ifdef EUCLIDEAN
	long double t0 = (*f).time[(*f).st_indx-1] + euclid_dist2d(st_point, (*f).nvp[(*f).st_indx-1])/(*f).vel[(*f).st_indx-1];	
	#else
	long double t0 = (*f).time[(*f).st_indx-1] + haversine_distance(st_point, (*f).nvp[(*f).st_indx-1])/(*f).vel[(*f).st_indx-1];
	#endif
	int n_nvp=(*f).n_nvp;
	long double **pos=(*f).pos;
	int st_indx=(*f).st_indx;
	long double *vel=(*f).vel;
	long double **nvp=(*f).nvp;
	
	if((*f).tp==0){
		pos[0][0]=st_point[0];
		pos[0][1]=st_point[1];
		pos[0][2]=st_point[2];
		pos[0][3]=st_point[3];
		pos[0][4]=st_point[4];
	}
	else{
		pos[0][0]=SAFE;
		pos[0][1]=SAFE;
		pos[0][2]=SAFE;
		pos[0][3]=0;
		pos[0][4]=0;
		for(j=0;j<DPOS;j++) pos[(*f).tp][j]=(*f).st_point[j];		
	}
	#ifdef EUCLIDEAN
	l_nvp = euclid_dist2d(st_point,nvp[st_indx]);
	#else
	l_nvp = haversine_distance(st_point,nvp[st_indx]);
	#endif
	// if t is smaller than t_wind*t_r ,the forecast on position is perfect
	// because otherwise there could be some actual conflicts.
	for(i=0;i<((*f).tp-1);i++) {
		for(j=0;j<DPOS;j++) pos[i+1][j]=SAFE;
		pos[i+1][3]=0;
		pos[i+1][4]=0;
	}
	
	int i_perf_fore = (int)(t_wind*t_r-1);
	for(i=(*f).tp,j=(st_indx-1),d=0 ; i<i_perf_fore ; i++){
		
		d+=vel[j]*time_step;
		if(d>l_nvp){
			d=d-l_nvp;
			t_c=d/vel[j];
			j++;
			(*f).time[j]=t0+(i+1)*time_step - t_c;

			if(j>(n_nvp-2)){
				for(;i<(t_wind-1);i++){
                    pos[i+1][0]=SAFE;
                    pos[i+1][1]=SAFE;
                    pos[i+1][2]=SAFE;
                    pos[i+1][3]=0;
					pos[i+1][4]=0;
				}
                return (0);
			}
			d=vel[j]*t_c;
			
			#ifdef EUCLIDEAN
			l_nvp = euclid_dist2d(nvp[j],nvp[j+1]);
			#else
			l_nvp=haversine_distance(nvp[j],nvp[j+1]);
			#endif
			
			if(!_nvp_to_close(&d,&t_c,vel,0.,nvp,&l_nvp,&j,n_nvp,(*f).time,t0+(i+1)*time_step)) {
				for(;i<(t_wind-1);i++){	
					pos[i+1][0]=SAFE;
					pos[i+1][1]=SAFE;
					pos[i+1][2]=SAFE;
					pos[i+1][3]=0;
					pos[i+1][4]=0;

				}
				return (0);
			}
			#ifdef EUCLIDEAN
			eucl_coord(vel[j], nvp[j], nvp[j+1], pos[i+1], t_c);
			#else
			coord(vel[j], nvp[j], nvp[j+1], pos[i+1], t_c);
			#endif
			pos[i+1][2]=nvp[j][2];
			pos[i+1][3]=nvp[j][3];
			pos[i+1][4]=nvp[j][4];
			
		}
		else {
			#ifdef EUCLIDEAN
			eucl_coord(vel[j], pos[i], nvp[j+1], pos[i+1], time_step);
			#else
		    coord(vel[j], pos[i], nvp[j+1], pos[i+1], time_step);
		    #endif
			pos[i+1][2]=nvp[j][2];
            pos[i+1][3]=nvp[j][3];
			pos[i+1][4]=nvp[j][4];
		}
	}

	//Here there is an error dV (fraction) on velocities. Note that in reality the 
	// aircraft is always going with the same speed, i.e. the noise is on the
	// forecast, not on the real speed.
	for(i=i_perf_fore;i<(t_wind-1);i++){
		d+=(vel[j]*(1. + dV))*time_step;
		if(d>l_nvp){
			d=d-l_nvp;
			t_c=d/(vel[j]*(1. + dV));
			j++;
			(*f).time[j]=t0+(i+1)*time_step - t_c;

			if(j>(n_nvp-2)){
				for(;i<(t_wind-1);i++){
                    pos[i+1][0]=SAFE;
                    pos[i+1][1]=SAFE;
                    pos[i+1][2]=SAFE;
                    pos[i+1][3]=0;
					pos[i+1][4]=0;
				}
                return (0);
			}
			d=(vel[j]*(1. + dV))*t_c;
			
			#ifdef EUCLIDEAN
			l_nvp=euclid_dist2d(nvp[j],nvp[j+1]);
			#else
			l_nvp=haversine_distance(nvp[j],nvp[j+1]);
			#endif
			
			if(!_nvp_to_close(&d,&t_c,vel,dV,nvp,&l_nvp,&j,n_nvp,(*f).time,t0+(i+1)*time_step)) {
				for(;i<(t_wind-1);i++){	
					pos[i+1][0]=SAFE;
					pos[i+1][1]=SAFE;
					pos[i+1][2]=SAFE;
					pos[i+1][3]=0;
					pos[i+1][4]=0;
				}
				return (0);
			}
			#ifdef EUCLIDEAN
			eucl_coord(vel[j]*(1. + dV), nvp[j], nvp[j+1], pos[i+1], t_c);
			#else
			coord(vel[j]*(1. + dV), nvp[j], nvp[j+1], pos[i+1], t_c);
			#endif
			
			pos[i+1][2]=nvp[j][2];
			pos[i+1][3]=nvp[j][3];
			pos[i+1][4]=nvp[j][4];
			
		}
		else {
			#ifdef EUCLIDEAN
			eucl_coord(vel[j]*(1. + dV), pos[i], nvp[j+1], pos[i+1], time_step);
			#else
		    coord(vel[j]*(1. + dV), pos[i], nvp[j+1], pos[i+1], time_step);
		    #endif
		    
			pos[i+1][2]=nvp[j][2];
            pos[i+1][3]=nvp[j][3];
			pos[i+1][4]=nvp[j][4];
		}
	}

	return 1;
}

/* Extract value from a Laplacian distribution*/
long double _extract_from_dist_laplace(long double mu,long double b){
	long double p=frand(-0.4999,0.5);
	return mu - b*(p/fabs(p))*log(1-2*fabs(p));
}

/*Modify the velocity of aircraft inside the sector according to a Laplacian distribution*/
int _laplacian_change_vel(Aircraft_t **f,int N_f){
	int i,j;
	long double M;
	for(i=0;i<N_f;i++){
		M=_extract_from_dist_laplace(MU ,B_LP);
		for(j=0;j<((*f)[i].n_nvp-1);j++) (*f)[i].vel[j]=(*f)[i].vel[j]+(*f)[i].vel[j]*M;
	}
	return 1;
}

/*Sort in the first position the element of that are flying (ready==1)*/
//~ int _sort_ready(Aircraft_t **f,int N_f){
	//~ int i,j;
	//~ Aircraft_t comodo;
	//~ 
	//~ for(i=0;i<N_f;i++){
		//~ if((*f)[i].ready) continue;
		//~ else for(j=i+1;j<N_f;j++) if((*f)[j].ready){
			//~ comodo=(*f)[i];
			//~ (*f)[i]=(*f)[j];
			//~ (*f)[j]=comodo;
		//~ }
	//~ }
	//~ return 1;
//~ }

int _sort_ready(Aircraft_t **f,TOOL_f *tl, int N_f){
	int i;
	
	int i_r = 0;
	int i_u = N_f-1;
	
	for(i=0;i<N_f;i++){
		if((*f)[i].ready){
			(*tl).F[i_r] = (*f)[i];
			i_r++;
		}
		else{
			(*tl).F[i_u] = (*f)[i];
			i_u--;			
			
		}
	}
	for(i=0;i<N_f;i++) (*f)[i] = (*tl).F[i];
	
	
	return 1;
}

int _set_st_point(Aircraft_t **f,int N_f,CONF_t conf){
	
	int i,j,old_st;
	
	long double d,t;
	/*The real position for departures*/
	int t_x = (conf.t_w*conf.t_r);
	int t_curr = (t_x-1)*conf.t_i;
	
	for(i=0;i<N_f;i++)if((*f)[i].ready){
		if((*f)[i].pos[t_x-1][0]==SAFE){
			(*f)[i].ready=0;
			continue;
		}
		for(j=0;j<DPOS;j++) (*f)[i].st_point[j] = (*f)[i].pos[t_x-1][j];
		
		old_st=(*f)[i].st_indx;
		
		if((*f)[i].st_point[0]==SAFE) {
			(*f)[i].st_indx=(*f)[i].n_nvp;
			(*f)[i].ready = 0;
			continue;

		}
		
		#ifdef EUCLIDEAN
		d = euclid_dist2d((*f)[i].pos[(*f)[i].tp], (*f)[i].nvp[old_st]);
		#else
		d = haversine_distance((*f)[i].pos[(*f)[i].tp], (*f)[i].nvp[old_st]);
		#endif
		t = (*f)[i].tp*conf.t_i + d / (*f)[i].vel[old_st-1];
		if(t>=t_curr) (*f)[i].st_indx = old_st;
		else{
			for(j=old_st;j< ((*f)[i].n_nvp-1);j++){
				#ifdef EUCLIDEAN
				d = euclid_dist2d((*f)[i].nvp[j], (*f)[i].nvp[j+1]);
				#else
				d = haversine_distance((*f)[i].nvp[j], (*f)[i].nvp[j+1]);
				#endif
				t +=  d / (*f)[i].vel[j];
		
				if(t>=t_curr) {
					(*f)[i].st_indx = j+1;
					break;
				}		
			}
			
		}
		
		if((*f)[i].st_indx>((*f)[i].n_nvp-1)) {
			(*f)[i].ready=0;
		}
	}
	return 1;
}


int _evaluate_neigh(Aircraft_t **f, int n_f,TOOL_f *tl,CONF_t conf){
	int i,j;
	
	for(i=0;i<n_f;i++) (*tl).n_neigh[i] = 0;
	
	//~ for(i=0;i<n_f;i++){
		//~ for(j=i+1;j<n_f;j++){
			//~ //if(haversine_distance((*f)[i].st_point,(*f)[j].st_point)<conf.d_neigh){
			//~ (*tl).neigh[j][(*tl).n_neigh[j]]=i;
			//~ ((*tl).n_neigh[j])+=1;
			//~ //}
		//~ }
	//~ }
	
	//~ for(i=0;i<n_f;i++) {
		//~ printf("%d:\t",i);
		//~ for(j=0;j<(*tl).n_neigh[i];j++) printf("%d\t",(*tl).neigh[i][j]);
		//~ printf("\n");
	//~ }
		
	

	for(i=0;i<n_f;i++){
		//printf("%Lf\t%Lf\n",(*f)[i].st_point[0],(*f)[i].st_point[1]);
		for(j=i+1;j<n_f;j++){
			#ifdef EUCLIDEAN
			if(euclid_dist2d((*f)[i].st_point,(*f)[j].st_point)<conf.d_neigh){
			#else
			if(haversine_distance((*f)[i].st_point,(*f)[j].st_point)<conf.d_neigh){
			#endif
				(*tl).neigh[j][(*tl).n_neigh[j]]=i;
				((*tl).n_neigh[j])+=1;
			}
		}
	}
		
	return 1;
	
}
int  _calculate_longest_direct(Aircraft_t *f,TOOL_f tl,CONF_t conf,int rer){
	int i;
	int plus;
	if(rer==1)  plus = 1;
	else plus = 2;
	
	
	int st_in = (int) (*f).pos[1][4];
	int sect;
	for(i=(*f).st_indx+1;i<(*f).n_nvp-plus;i++) {
		sect = (int) (*f).nvp[i][4];
		if(st_in != sect ) if((conf.capacy[sect])<tl.workload[sect]){
			if (rer) printf("rer Overfull Capacity %d \t %d -> %d\t%d\n",(conf.capacy[sect]),tl.workload[sect],st_in,i-1);
			else  printf("dir Overfull Capacity %d \t %d -> %d\t%d\n",(conf.capacy[sect]),tl.workload[sect],st_in,i-1);
			
			//return (*f).n_nvp-plus;

			return i-1;
		}
	}

	return (*f).n_nvp-plus;
}	
			

int _calculate_st_point(Aircraft_t *f,CONF_t conf,long double t){
	(*f).st_indx=1;

	if(!_position(f, (*f).nvp[0], conf.t_w, conf.t_i, conf.t_r, 0.)){
		(*f).ready=0;
		return (0);
	}

	
	/*Se real position for departures*/
	int t_x = (int) ( (*f).time[0] - (t - (conf.t_i*(conf.t_w*conf.t_r-1))))/conf.t_i ;
	
	if(t_x>=conf.t_w||t_x<0){
		printf("BOU %d\n",t_x);
	}
	
	(*f).st_point[0]=(*f).pos[t_x][0];
    (*f).st_point[1]=(*f).pos[t_x][1];
	(*f).st_point[2]=(*f).pos[t_x][2];
    (*f).st_point[3]=(*f).pos[t_x][3];
	
	(*f).st_indx = find_st_indx((*f).nvp, (*f).n_nvp, (*f).st_point,0,(*f).st_point);
	//printf("%d\n", (*f).st_indx);
		
#ifdef DEBUG0
	if((*f).st_indx ==-1) {
		printf("%Lf\t%Lf\n",(*f).st_point[0],(*f).st_point[1]);
		//plot_pos((*f),conf);
		plot_where((*f),conf,"/tmp/nvp_f");
	//if(1){
		printf("st: %Lf\t%Lf\n",(*f).st_point[0],(*f).st_point[1]);
		printf("%Lf\n",distance_point_segment((*f).nvp[1], (*f).nvp[2], (*f).pos[conf.t_w-1]));
		
		printf("%d st_indx\n",(*f).st_indx);
		int i;
		for(i=0;i<conf.t_w;i++) printf("%Lf\t%Lf\n",(*f).pos[i][0],(*f).pos[i][1]);
		//plot_pos((*f), conf);
		printf("ID: %d\n",(*f).ID);
		BuG("Error on st_indx on Departures\n");
	}
#endif
	
	return 1;
}

/* Put to one ready flag for plane on departures and sort in the first position them 
 and calculate starting point */
int _get_ready(Aircraft_t **f, TOOL_f *tl, int N_f,long double t,CONF_t conf){
	int i,j;
	/* Time step */
	long double t_stp=(conf.t_r*conf.t_w*conf.t_i);

	/* For each flight */
	for(i=0;i<N_f;i++) {
		/* if it is not fliyng*/
		if((*f)[i].ready==0) {
			/*if it has to departure in the current time-step*/
			if( (*f)[i].time[0]>=t&& (*f)[i].time[0]< (t+t_stp) )  {
				 // set ready flag
				(*f)[i].ready=1;
				 /*evalute the shift from the first time-increment of the current time-step*/
				(*f)[i].tp = ((*f)[i].time[0]-t)/conf.t_i;
				/*Set the pointer to the second nvp of the flight*/
				(*f)[i].st_indx = 1;
				/*set the stating position to the fist nvp of the flight*/
				for(j=0;j<DPOS;j++) (*f)[i].st_point[j] = (*f)[i].nvp[0][j];
			}
		}
		else (*f)[i].tp = 0;
	}
	
	/*Evalute the number of active flight*/
	int n_f=0;
	for(i=0;i<N_f;i++) if((*f)[i].ready==1) n_f++;
	
	/* sort in the fist position of the Aircraft array the active flight*/
	_sort_ready(f,tl,N_f);
	
	return n_f;
}

int _mix_flight(Aircraft_t **x,int *list,int N){
	int i;
	Aircraft_t comodo;
	
	for(i=0;i<(N-1);i++){
		comodo=(*x)[list[i]];
		(*x)[list[i]]=(*x)[i];
		(*x)[i]=comodo;
	}
	
	return 1;
}

int _suffle_list(Aircraft_t **f, int n, TOOL_f *tool){
	int i,j;
	Aircraft_t t;
	
	if (n > 1) {
 
        for (i = 0; i < n - 1; i++) {
          j = i + rand() / (RAND_MAX / (n - i) + 1);
          t = (*f)[j];
          (*f)[j] = (*f)[i];
          (*f)[i] = t;
        }
    }
	
	return 1;
}


int _suffle_list_top(Aircraft_t **f, int n_f, TOOL_f *tool){
	int i;
		
	(*f)[n_f].touched = 0;
	
	
	int i_pre = 0;
	int i_post = n_f-1;
	for(i=0;i<n_f;i++){
		if((*f)[i].touched == 1 ){
			(*tool).F[i_pre] = (*f)[i];
			i_pre++;
		}
	}
	(*tool).F[i_pre]=(*f)[n_f];
	for(i=0,++i_pre ;i<n_f;i++) if( (*f)[i].touched == 0 ){
		(*tool).F[i_pre] = (*f)[i];
		i_pre++;
	}
		
	for(i=0;i<=n_f;i++) (*f)[i] = (*tool).F[i];	
		
	return 1;
}

int _excange_in_list(Aircraft_t **f,int n_f,TOOL_f *tl){
	
	int sel=irand(n_f);
	sel=0;
	printf("%d\tn %d\n",(*tl).lista[n_f],(*f)[n_f].n_nvp);
	int comodo=(*tl).lista[n_f];
	int i;
	//for(i=0;i<n_f;i++) tool.lista[i]=i;
	
	for(i=n_f;i>(sel);i--) (*tl).lista[i]= (*tl).lista[i-1];
	(*tl).lista[sel]=comodo;
	_mix_flight(f,(*tl).lista,n_f);
	return 1;

}

/*Check collision in flight distance*/
// TODO: take safety events indices from _check_risk
int _checkFlightsCollision(long double *d,CONF_t conf,Aircraft_t *f){
	int i,indx;
	
	for(indx=1;indx<conf.t_w;indx++) if (d[indx]<(conf.d_thr*(1+_f_dist(indx,&conf) )) )  break;
	if(indx==conf.t_w) return 0;

	/*return navpoint immediatly after the collision*/
	#ifdef EUCLIDEAN
	if(eucl_isbetween((*f).st_point, (*f).nvp[(*f).st_indx], (*f).pos[indx])) return (*f).st_indx;
	#else
	if(isbetween((*f).st_point, (*f).nvp[(*f).st_indx], (*f).pos[indx])) return (*f).st_indx;	
	#endif
	for(i=(*f).st_indx ; i<((*f).n_nvp-1);i++){
		#ifdef EUCLIDEAN
		if(eucl_isbetween((*f).nvp[i],(*f).nvp[i+1],(*f).pos[indx])) {
		#else
		if(isbetween((*f).nvp[i],(*f).nvp[i+1],(*f).pos[indx])) {
		#endif
		    return i+1;}
	}
	/*if it print this is a BUG*/
	printf("%d\t%Lf\t%Lf\n",indx,(*f).pos[indx][0],(*f).pos[indx][1]);
	printf("\n");
	for(i=0;i<conf.t_w;i++) printf("%d]\t%Lf\t%Lf\t%d\n",i,d[i],(conf.d_thr*(1+(i-(*f).tp)*conf.noise_d_thr)),d[i]<(conf.d_thr*(1+(i-1)*conf.noise_d_thr)));
	return -1;
}

int _minimum_flight_distance(long double **pos,Aircraft_t **f,int N_f,int N,TOOL_f tl){
	int h,i,j;
	long double min;
	
	for(i=0;i<N;i++){
		if((int)pos[i][3]&&N_f>0) {
			/*if the flight are active and on the same flight level*/
			if(((int)pos[i][2])!=((int)(*f)[0].pos[i][2])||((int) (*f)[0].pos[i][3])==0) min=SAFE;
			#ifdef EUCLIDEAN
			else  min=euclid_dist2d(pos[i],(*f)[0].pos[i]);
			#else
			else  min=haversine_distance(pos[i],(*f)[0].pos[i]);
			#endif
			
			/*It checks for the flight in the edgelist*/
			for(h=0;h<tl.n_neigh[N_f];h++){
				j=tl.neigh[N_f][h];
				
				if( fabsl((int)pos[i][2]-((int)(*f)[j].pos[i][2]))<9.99&&(int) (*f)[j].pos[i][3]==1){
					#ifdef EUCLIDEAN
					tl.dist[i]=euclid_dist2d(pos[i],(*f)[j].pos[i]);					
					#else
					tl.dist[i]=haversine_distance(pos[i],(*f)[j].pos[i]);
					#endif
					if(tl.dist[i]<min) min=tl.dist[i];
				}
			}
			tl.dist[i]=min;
		}
		else tl.dist[i]=SAFE;
	}

	return 1;	
}

int _isInsideCircle(long double *p,long double *shock){
	#ifdef EUCLIDEAN
	return euclid_dist2d(p,shock)<shock[2];
	#else
	return haversine_distance(p,shock)<shock[2];
	#endif
}

int _checkShockareaRoute(long double **pos,int N,SHOCK_t shock,long double *d,long double t){
	int i,j;
	
	for(j=0;j<shock.Nshock;j++) if( fabs(shock.shock[j][3]-t)<SGL ){
			for(i=1;i<N;i++) if(pos[i][3]==1) if(pos[i][2]==shock.shock[j][5]){
				 if(_isInsideCircle(pos[i],shock.shock[j])){
					 d[i]=0;
				}
		}
	}
	
	return 0;
}

//TODO: record indices.
int _check_risk(long double *d,CONF_t conf,int t_w,int tp){
	int i,risk;
	for(i=1,risk=0;i<t_w;i++) if(d[i]!=SAFE) {
		if(d[i]<=(conf.d_thr*(1+_f_dist(i,&conf) )) ) risk=1;
	}
	
	return risk;
}


int _get_d_neigh(CONF_t *conf,Aircraft_t **f,int N_f){
	int i,j;
	long double max_v=0;
	for(i=0;i<N_f;i++) for(j=0;j<((*f)[i].n_nvp-1);j++) if((*f)[i].vel[j]>max_v) max_v = (*f)[i].vel[j];
	
	(*conf).d_neigh = 2.*(2.*((*conf).t_w*(*conf).t_i)*max_v*(1+(*conf).sig_V)) + ((*conf).d_thr*(1+_f_dist(2.*(*conf).t_w,conf))) ;
	
	return 1;
	
}

//int _add_nvp_inSector(Aircraft_t *f,int safe){
/*
int _add_nvp_inSector(Aircraft_t *f){	
	int i;
	long double *st_nvp=falloc_vec(DPOS);
	
	for(i=0;i<DPOS;i++) st_nvp[i]=(*f).pos[0][i];
	add_nvp(f, st_nvp);
	//((*f).bound[1])++;
	//((*f).st_indx)++;
	
	return 1;
}*/

int _temp_angle_nvp(Aircraft_t *f,CONF_t conf, TOOL_f *tl,int safe){
	int i;
	#ifdef EUCLIDEAN
	for(i=0;i<conf.n_tmp_nvp;i++) if(euclid_dist2d((*f).st_point,conf.tmp_nvp[i])<DTMP_P){
	#else
	for(i=0;i<conf.n_tmp_nvp;i++) if(haversine_distance((*f).st_point,conf.tmp_nvp[i])<DTMP_P){	
	#endif
	#ifdef EUCLIDEAN
		(*tl).temp_angle[i]=acosl(cosl(fabsl(eucl_angle_direction( (*f).nvp[safe],(*f).st_point, conf.tmp_nvp[i]))));		
	#else
		(*tl).temp_angle[i]= PI - angle_direction( (*f).nvp[safe],(*f).st_point, conf.tmp_nvp[i]);
	#endif 
	}

	return 1;
}

int _select_candidate_tmp_nvp_shortpath(Aircraft_t *f,CONF_t conf, TOOL_f *tl, int safe){
	int i;
	
	_temp_angle_nvp(f,conf,tl,(*f).st_indx);
#ifdef EUCLIDEAN
	for(i=0,(*tl).n_sel_nvp=0 ;i<conf.n_tmp_nvp;i++) if(euclid_dist2d((*f).st_point,conf.tmp_nvp[i])<DTMP_P) if((*tl).temp_angle[i]<conf.max_ang) if(frand(0,100)<50){
		(*tl).sel_nvp_index[(*tl).n_sel_nvp][0]= euclid_dist2d((*f).st_point, (conf).tmp_nvp[i])+euclid_dist2d( (conf).tmp_nvp[i],(*f).nvp[safe+1]);
#else
	for(i=0,(*tl).n_sel_nvp=0 ;i<conf.n_tmp_nvp;i++) if(haversine_distance((*f).st_point,conf.tmp_nvp[i])<DTMP_P) if((*tl).temp_angle[i]<conf.max_ang) if(frand(0,100)<50){

		(*tl).sel_nvp_index[(*tl).n_sel_nvp][0]= haversine_distance((*f).st_point, (conf).tmp_nvp[i])+haversine_distance( (conf).tmp_nvp[i],(*f).nvp[safe+1]);
#endif
		(*tl).sel_nvp_index[(*tl).n_sel_nvp][1]=i;
		((*tl).n_sel_nvp)++;
	}

	q_sort((*tl).sel_nvp_index, 0, (*tl).n_sel_nvp);
	
	return 1;
}

int _select_candidate_tmp_nvp(Aircraft_t *f,CONF_t conf, TOOL_f *tl, int safe){
	int i;
	
	_temp_angle_nvp(f,conf,tl,safe);
	for(i=0,(*tl).n_sel_nvp=0 ;i<conf.n_tmp_nvp;i++) if((*tl).temp_angle[i]<conf.max_ang){

		(*tl).sel_nvp_index[(*tl).n_sel_nvp][0]=(*tl).temp_angle[i];
		(*tl).sel_nvp_index[(*tl).n_sel_nvp][1]=i;
		((*tl).n_sel_nvp)++;
	}
	
	q_sort((*tl).sel_nvp_index, 0, (*tl).n_sel_nvp);

	return 1;
}


int _temp_new_nvp(Aircraft_t *f, int safe, Aircraft_t *new_f,CONF_t conf){
	int i;
	
	
	(*new_f).nvp = (long double**) malloc( (*new_f).n_nvp *sizeof(long double *));
	if((*new_f).nvp==NULL) BuG("not enought Memory\n");
	
	(*new_f).pos=falloc_matrix(conf.t_w, DPOS);
	
	/*Navigation Point*/
	int h;
	for(i=0;i<safe;i++) (*new_f).nvp[i]=(*f).nvp[i];
	long double *p=(long double*) malloc(DPOS*sizeof(long double));
	(*new_f).nvp[i]=p;
	h=i+2;
	if(h>=(*f).n_nvp) {
		h--;
	}
	for(++i;h<(*f).n_nvp;h++,i++) (*new_f).nvp[i]= (*f).nvp[h];
	
	/*Velocity*/
	(*new_f).vel=falloc_vec((*new_f).n_nvp-1);
	for(i=0;i<=(safe-1);i++) (*new_f).vel[i]=(*f).vel[i];
	(*new_f).vel[i]=(*f).vel[i-1];
	h=i+2;
	if(h>=(*f).n_nvp) h--;
	for(++i;h<((*f).n_nvp-1);i++,h++) {
		if(i>=(*new_f).n_nvp-1) BuG("Ex\n");
		(*new_f).vel[i]=(*f).vel[h];
	}
	
	(*new_f).st_indx=(*f).st_indx;
	
	if((safe+2)>=(*f).n_nvp) (*new_f).bound[1]=(*f).bound[1];
	else (*new_f).bound[1]=(*f).bound[1]-1;
		
	return 1;
}

int _reroute(Aircraft_t *f,Aircraft_t *flight,int N_f,SHOCK_t sh,CONF_t conf, TOOL_f tl,int unsafe,long double t){
	

	long double dV = tl.dV[N_f]; // N_F because we test rerouting against all previous flights.

	/*Select the candidate temp-nvp according to the angle constrains. 
	 * 1)The list is sorted according to shorted-path
	 * 2) the list is sorted according to smallest angle*/
	if(conf.shortest_path) _select_candidate_tmp_nvp_shortpath(f,conf,&tl,(*f).st_indx);
	else 	_select_candidate_tmp_nvp(f,conf,&tl,(*f).st_indx);
	
	int i,j;
	int n_old=unsafe - (*f).st_indx + 1;
			
	/*create a backup for the velocity*/
	long double *old_vel=falloc_vec((*f).n_nvp-1);
	for(i=0;i<((*f).n_nvp-1);i++) old_vel[i]=(*f).vel[i];
	
	/*Evalute the mean velocity for the skipped segment*/
	long double vm=mean(&((*f).vel[(*f).st_indx-1]),unsafe-(*f).st_indx+2);
	(*f).vel[(*f).st_indx-1]=vm;
	(*f).vel[(*f).st_indx]=vm;
	
	for(i=(*f).st_indx+1;i<((*f).n_nvp-2);i++) (*f).vel[i]=(*f).vel[i+1];

	/*create a backup for the old nvp*/
	long double **old_nvp=falloc_matrix((*f).n_nvp,DPOS);
	int olf=(*f).n_nvp;
	for(i=0;i<(*f).n_nvp;i++) for(j=0;j<DPOS;j++) old_nvp[i][j]=(*f).nvp[i][j];
		
	int rp_temp;
	
	for(i=unsafe+1,rp_temp=0;i<(*f).n_nvp;i++,rp_temp++) {
		for(j=0;j<DPOS;j++) (*f).nvp[i-n_old+1][j]=(*f).nvp[i][j];
	}
	
	(*f).n_nvp=(*f).n_nvp -  n_old+1;
	/*Evaluate the longest possible direct 
	 * according to the capacity constrain*/
	int longest_rer =  _calculate_longest_direct(f,tl,conf,1);
	if (longest_rer< ((*f).n_nvp-1) ) printf("%d\t%d\n",longest_rer,((*f).n_nvp-1) );


	int solved,h,dj,l,m;
	//double newv;
	long double temp_rout;

	/*For every comeback point after collision*/
	for(dj=0,j=((*f).st_indx+1);j<longest_rer;j++,dj++){
		
		/* The first time 0 the other 1*/
		dj = (dj > 0);
		
		/*Modify trajectory reducing point*/
		for(h=((*f).st_indx+1),m=0;h<((*f).n_nvp-dj);h++,m++) {
			for(l=0;l<DPOS;l++) (*f).nvp[h][l]=(*f).nvp[(*f).st_indx+dj+m+1][l];
			
			(*f).vel[h-1]=(*f).vel[(*f).st_indx+dj+m];
		}

		(*f).n_nvp=(*f).n_nvp-dj;
		
		/*For each nvp in the candidate list of temp_nvp*/
		for(i=0;i<tl.n_sel_nvp;i++){
			
			(*f).nvp[(*f).st_indx][0]=conf.tmp_nvp[(int) tl.sel_nvp_index[i][1]][0];
			(*f).nvp[(*f).st_indx][1]=conf.tmp_nvp[(int) tl.sel_nvp_index[i][1]][1];
			(*f).nvp[(*f).st_indx][2]=old_nvp[unsafe][2];
			(*f).nvp[(*f).st_indx][3]=old_nvp[unsafe][3];
	
			#ifdef EUCLIDEAN
			temp_rout = euclid_dist2d((*f).st_point, (*f).nvp[(*f).st_indx])/(*f).vel[(*f).st_indx-1]+euclid_dist2d((*f).nvp[(*f).st_indx+1], (*f).nvp[(*f).st_indx])/(*f).vel[(*f).st_indx];
			#else
			temp_rout = haversine_distance((*f).st_point, (*f).nvp[(*f).st_indx])/(*f).vel[(*f).st_indx-1]+haversine_distance((*f).nvp[(*f).st_indx+1], (*f).nvp[(*f).st_indx])/(*f).vel[(*f).st_indx];
			#endif
			
			/* If the flight does not come back before 2*t_w */
			if(temp_rout>(conf.t_w*2*conf.t_i)) {
				continue;
			}
			/*if the comeback angle is less then max_ang*/
			#ifdef EUCLIDEAN
			if(acosl(cosl(fabsl(eucl_angle_direction((*f).nvp[(*f).st_indx],(*f).nvp[(*f).st_indx+1], (*f).st_point))))>conf.max_ang) continue;
			#else
			if((PI - angle_direction((*f).nvp[(*f).st_indx],(*f).nvp[(*f).st_indx+1], (*f).st_point))>conf.max_ang) continue;
			#endif

			/*cheak the possible solution*/
			_position(f , (*f).st_point , conf.t_w, conf.t_i, conf.t_r, dV);		
			_minimum_flight_distance((*f).pos,&flight,N_f,conf.t_w,tl);
			_checkShockareaRoute((*f).pos,conf.t_w, sh,tl.dist,t);
			solved=_check_risk(tl.dist,conf,conf.t_w,(*f).tp);
		
			/*if solved*/
			if(solved==0) {

				_position(f , (*f).st_point , conf.t_w*2, conf.t_i, conf.t_r, dV);
				ffree_2D(old_nvp, olf);
				free(old_vel);
				(*f).touched = 1;
				return 0;
			}
		}
	}

	/*NOT SOLVED*/
	for(i=0;i<olf;i++) for(j=0;j<DPOS;j++) (*f).nvp[i][j]=old_nvp[i][j];
	for(i=0;i<(olf-1);i++)  (*f).vel[i]=old_vel[i];
	
	(*f).n_nvp=olf;
	ffree_2D(old_nvp, olf);
	_position(f , (*f).st_point , conf.t_w, conf.t_i, conf.t_r, dV);
	free(old_vel);
	
	return 1;
}

int _try_flvl(Aircraft_t *f,Aircraft_t **flight,int N_f,CONF_t conf,TOOL_f tl,SHOCK_t sh, long double f_lvl_on,int safe,int end_p,long double t){
	int i;
	for(i=((*f).st_indx-1);i<end_p;i++) (*f).nvp[i][2]=f_lvl_on;
	(*f).st_point[2]=f_lvl_on;
	//_position(f, (*f).st_point, conf.t_w, conf.t_i);
	for(i=0;i<conf.t_w;i++) (*f).pos[i][2]=f_lvl_on;
	/* you dont't need to recompute the positions because it has been done in reroute.*/
	_minimum_flight_distance((*f).pos,flight,N_f,conf.t_w,tl);
	_checkShockareaRoute((*f).pos,conf.t_w, sh,tl.dist,t);
	safe=_check_risk(tl.dist,conf,conf.t_w,(*f).tp);
	return safe;
}

int _is_close_nvp(Aircraft_t f){
	#ifdef EUCLIDEAN
	if(euclid_dist2d( (f).nvp[(f).st_indx-1],(f).st_point)<500) return 0;
	#else 
	if(haversine_distance ( (f).nvp[(f).st_indx-1],(f).st_point)<500) return 0;	
	#endif
	else return 1;
}

int _find_end_point(Aircraft_t *f,CONF_t conf){
	
	int i,j;
	for(i=conf.t_w-1;i>0;i--) if((*f).pos[i][0]!=SAFE) break;
	
	
	
	long double t_curr = i * conf.t_i;
	long double t = (*f).tp * conf.t_i;
	long double d;
	
	#ifdef EUCLIDEAN
	d = euclid_dist2d ((*f).st_point,(*f).nvp[(*f).st_indx]);
	#else
	d = haversine_distance((*f).st_point,(*f).nvp[(*f).st_indx]);	
	#endif


	t+= d/(*f).vel[(*f).st_indx-1];

	
	if( t>=t_curr ) return (*f).st_indx;
	for( j = (*f).st_indx ; j<((*f).n_nvp-1) ; j++){
		#ifdef EUCLIDEAN
		d = euclid_dist2d((*f).nvp[j],(*f).nvp[j+1]);
		#else
		d = haversine_distance((*f).nvp[j],(*f).nvp[j+1]);	
		#endif
		t+= d/(*f).vel[j];
		if(t>=t_curr) return j+1;
	}
	return ((*f).n_nvp-1);
	
	//~ int end = find_p_indx((*f).nvp,(*f).n_nvp,(*f).pos[i]);
	//~ if (end==0) {
		//~ return (*f).st_indx;
		//~ //return (*f).n_nvp;
	//~ }
	//~ else return end;	
}

int _change_flvl(Aircraft_t *f,Aircraft_t **flight,int N_f,CONF_t conf,TOOL_f tl,SHOCK_t sh,long double t){
	int unsafe = (*f).st_indx-1;
	
	/* The flight will come back to the original flight level after 2*t_w (endp)*/
	int endp = _find_end_point(f,conf);
	
	long double *h=falloc_vec(endp-((*f).st_indx-1));
	int i,j;
	for(i=(*f).st_indx-1,j=0;i<endp;i++,j++) h[j]=(*f).nvp[i][2];
	
	long double f_lvl_on=(*f).nvp[  unsafe ][2];

	/*It tries first +20FL*/
	if(!_try_flvl(f,flight,N_f,conf,tl,sh,f_lvl_on+20.,unsafe,endp,t)){
		free(h);
		(*f).touched = 1;
		return 0;
	 }
	 /*if it does not work, ti tries -20FL*/
	if(!_try_flvl(f,flight,N_f,conf,tl,sh,f_lvl_on-20.,unsafe,endp,t)) {
		free(h);
		(*f).touched = 1;
		return 0;
	}
	
	/*if it does not work return 1*/
	for(i=(*f).st_indx-1,j=0;i<endp;i++,j++) (*f).nvp[i][2]=h[j];
	free(h);
	
	return 1;
}

double _calculate_optimum_direct(Aircraft_t *f,int on){
	#ifdef EUCLIDEAN
	double d1=euclid_dist2d((*f).st_point,(*f).nvp[(*f).st_indx+on]);
	double d2=euclid_dist2d((*f).st_point,(*f).nvp[(*f).st_indx]);
	int i;
	for(i=0;i<on;i++){
		d2+=euclid_dist2d((*f).nvp[(*f).st_indx+i],(*f).nvp[(*f).st_indx+i+1]);
	}
	#else
	double d1=haversine_distance((*f).st_point,(*f).nvp[(*f).st_indx+on]);
	double d2=haversine_distance((*f).st_point,(*f).nvp[(*f).st_indx]);
	int i;
	for(i=0;i<on;i++){
		d2+=haversine_distance((*f).nvp[(*f).st_indx+i],(*f).nvp[(*f).st_indx+i+1]);
	}	
	#endif
	return d2-d1;							 
}


int _direct(Aircraft_t *f,Aircraft_t *flight,int N_f,CONF_t conf, TOOL_f tl,SHOCK_t sh,long double tt){
	/*If the new path is less than the original*/
	long double diff[]={0,0};
	int old_st_indx=(*f).st_indx;
	int i=(*f).st_indx;
	

	int o;
	
	int j;
	#ifdef EUCLIDEAN		
	long double t=euclid_dist2d( (*f).pos[0], (*f).nvp[(*f).st_indx+1] )/(*f).vel[i];
	#else
	long double t=haversine_distance( (*f).pos[0], (*f).nvp[(*f).st_indx+1] )/(*f).vel[i];	
	#endif
	
	/*Evalue the improvent of jumping 1 nvp*/
	diff[0]=_calculate_optimum_direct(f, 1);
	/* If it smaller the LSKm exit*/
	if(diff[0]<LS) return 0;
		
	int h;
	/*it evalueat the longest possible direct 
	 * according to the capacity contrains of the sectors*/
	int longest_direct = _calculate_longest_direct(f,tl,conf,0);
	
	for(i=((*f).st_indx+1),h=1;i<(longest_direct) &&t<(3.5*conf.t_w*conf.t_i);i++,h++) {
		/*Evalue the improvent of jumping h+1 nvp*/
		diff[1]=_calculate_optimum_direct(f, h+1);
		/* If it smaller the LSKm it gives the previous direct*/
		if ((diff[1]-diff[0])<LS){
			i--;
			break;
		}
		
		diff[0]=diff[1];
		#ifdef EUCLIDEAN
		t=euclid_dist2d((*f).pos[0], (*f).nvp[i+1])/(*f).vel[i];
		#else
		t=haversine_distance((*f).pos[0], (*f).nvp[i+1])/(*f).vel[i];		
		#endif

	}
	int r;
	
	//(*f).st_indx=(*f).st_indx+h;
	int old_n_nvp = (*f).n_nvp;
	long double **old_nvp = falloc_matrix((*f).n_nvp,DPOS);
	long double *old_vel = falloc_vec((*f).n_nvp-1);
	
	/*COPY*/
	for(o=0;o<(old_n_nvp-1);o++) {
		for(j=0;j<DPOS;j++) old_nvp[o][j]=(*f).nvp[o][j];
		old_vel[o]=(*f).vel[o];
	}
	for(j=0;j<DPOS;j++) old_nvp[o][j]=(*f).nvp[o][j];
	
	for(r=old_st_indx;r<((*f).n_nvp-h);r++) {
		for(j=0;j<DPOS;j++) (*f).nvp[r][j]=(*f).nvp[r+h][j];
		(*f).vel[r-1]=(*f).vel[r+h-1];
	}

	(*f).n_nvp=(*f).n_nvp-h;

	/*Evaluate the risk on the new route*/
	_position(f, (*f).st_point, conf.t_w*2, conf.t_i, conf.t_r, 0.); //TODO: maybe we should change this.
	_minimum_flight_distance((*f).pos,&flight,N_f,2*conf.t_w,tl);
	_checkShockareaRoute((*f).pos,conf.t_w*2, sh,tl.dist,tt);
	
	//~ printf("A\t");
	//~ print_time(tt);

	
	/*if it will involved in a conflict exit witout a direct*/
	if(_check_risk(tl.dist,conf,conf.t_w*2,(*f).tp)){
		for(o=0;o<(old_n_nvp-1);o++) {
			for(j=0;j<DPOS;j++) (*f).nvp[o][j]=old_nvp[o][j];
			(*f).vel[o]=old_vel[o];
		}
		for(j=0;j<DPOS;j++) (*f).nvp[o][j]=old_nvp[o][j];
		
		(*f).n_nvp = old_n_nvp;
		_position(f, (*f).st_point, conf.t_w*2, conf.t_i, conf.t_r, 0.);
		ffree_2D(old_nvp,old_n_nvp);
		free(old_vel);

		return 0;
	}
	/*Outwise the direct is accepted*/
	else{
		
		_position(f, (*f).st_point, conf.t_w*2, conf.t_i, conf.t_r, 0.);

	}
	//~ printf("D\t");
	//~ print_time(tt);
	
	ffree_2D(old_nvp,old_n_nvp);
	free(old_vel);
	(*f).touched = 1; //touch the flight
	
	return 1;
}

int _check_safe_events(Aircraft_t **f, int N_f, SHOCK_t sh,TOOL_f tl, CONF_t conf, long double t){
	
	int i,unsafe=0;

	/*For each flight the are flying*/
	for(i=1;i<N_f;i++) if((*f)[i].ready){
		
		/*Evalute the minimum distance array for the i-flight*/
		_minimum_flight_distance((*f)[i].pos,f,i,conf.t_w,tl);
		
		/*Modify the minimum distance array with a zero if it will cross a shock*/
		_checkShockareaRoute((*f)[i].pos,conf.t_w, sh,tl.dist,t);
		/*check if the minimu  distance array has some value under the safaty distance*/
		//unsafe=_check_risk(tl.dist,conf,conf.t_w,(*f)[i].tp);
		
		if(unsafe) {
			/* If I already moved the i-Flight in the current time step*/
			#ifdef SINGLE_TOUCH
			if( (*f)[i].touched == 1 ) return i;
			#endif
			
			/*Return the next nvp after the first expected collision*/
			unsafe=_checkFlightsCollision(tl.dist, conf, &(*f)[i]); 

			/*try to reroute the flight*/
			unsafe=_reroute(&((*f)[i]),(*f),i,sh,conf,tl,unsafe,t);
			
			/*If it does not work*/
			if(unsafe) 
				/*try to chenge the flight level*/
				unsafe=_change_flvl(&(*f)[i],f,i,conf,tl,sh,t);
	
			if(unsafe) {
				/*if it does not work*/
				//~ printf("Unsolved %d Flight\n",(*f)[i].ID);
				//~ print_time(t);
				//~ printf("\n");
				return i;
			}
			
		}
		/* If no conflict are detected it tries with a prob direct_thr to give a direct*/
		else if(frand(0,1)<conf.direct_thr) if((*f)[i].st_indx< ((*f)[i].n_nvp-2)&&(*f)[i].st_indx>1) {
			//add_nvp_st_pt(&(*f)[i]);
			_direct(&(*f)[i],(*f),i,conf,tl,sh,t);
		}
		 
	}
	
	
	return -1;
}



int _expected_fly(Aircraft_t **f, int N_f, CONF_t conf, TOOL_f tl){
	int i;
	for(i=0;i<N_f;i++){
		/* Compute who is ready*/
		if((*f)[i].ready){
			/*Compute for 2*t_w*/
			if(!_position(&(*f)[i], (*f)[i].st_point, conf.t_w*2, conf.t_i, conf.t_r, tl.dV[i]) && (*f)[i].pos[0][0]==SAFE) ;
		}
	}
	return 1;
}

int _draw_dV(int N_f, CONF_t conf, TOOL_f *tl){
	int i;
	for(i=0;i<N_f;i++){
		(*tl).dV[i] = frand(-conf.sig_V, conf.sig_V);	
	}

	return 1;
}

int _evaluate_workload(Aircraft_t **f,int n_f,int N_f,TOOL_f tl,CONF_t conf, long double curr_t){
	int i,j,t;
	
	/*For the Flight that are flying */
	
	int walk[(conf).n_sect];
	
	for(i=0;i<conf.n_sect;i++) tl.workload[i]=0;

	#ifdef CAPACITY
	
	for(i=0 ;i<n_f;i++) {
		
		for(j=0;j<(conf).n_sect;j++) walk[j]=0;
		#ifdef EUCLIDEAN
		t = euclid_dist2d((*f)[i].st_point,(*f)[i].nvp[(*f)[i].st_indx])/(*f)[i].vel[(*f)[i].st_indx-1];
		#else
		t = haversine_distance((*f)[i].st_point,(*f)[i].nvp[(*f)[i].st_indx])/(*f)[i].vel[(*f)[i].st_indx-1];		
		#endif
		( walk[ (int) (*f)[i].pos[1][4]] )++;
		
		for(j=(*f)[i].st_indx;j<((*f)[i].n_nvp-1);j++) {
			#ifdef EUCLIDEAN
			t += euclid_dist2d((*f)[i].nvp[j],(*f)[i].nvp[j+1])/(*f)[i].vel[j];			
			#else
			t += haversine_distance((*f)[i].nvp[j],(*f)[i].nvp[j+1])/(*f)[i].vel[j];
			#endif
			if( t > 3600. ) break;
			
			(walk[(int) (*f)[i].nvp[j][4]])++;
		}
		for(j=1;j<(conf).n_sect;j++) if(walk[j]!=0) (tl.workload[j])++;	
	}
	
	/*For the Flight that will fly */
	for(;i<N_f;i++) if( (curr_t < (*f)[i].time[0]) && (curr_t +3600 > (*f)[i].time[0]) )  {
		for(j=0;j<(conf).n_sect;j++) walk[j]=0;
		
		t = curr_t - (*f)[i].time[0];
		for(j=0;j<((*f)[i].n_nvp-1);j++){
			#ifdef EUCLIDEAN
			t += euclid_dist2d((*f)[i].nvp[j],(*f)[i].nvp[j+1])/(*f)[i].vel[j];
			#else
			t += haversine_distance((*f)[i].nvp[j],(*f)[i].nvp[j+1])/(*f)[i].vel[j];			
			#endif
			if( t > 3600. ) break;
			(walk[(int) (*f)[i].nvp[j][4]])++;	
		}
		for(j=1;j<(conf).n_sect;j++) if(walk[j]!=0) (tl.workload[j])++;	
	}
	#endif
	return 0;
}


void print_workload(TOOL_f tl, CONF_t conf, char *file_w){
	int i,j;
	
	FILE *wstream = fopen(file_w,"a");
	if(wstream==NULL) BuG("Impossible to write workload\n");
	
	
	for(j=1;j<conf.n_sect;j++) fprintf(wstream, "%d\t",tl.workload[j]);
	fprintf(wstream, "\n");

	fclose(wstream);
	return;
	
}

int _update_shock(SHOCK_t *sh,int t,CONF_t *conf){
	int i;
	for(i=0;i<(*sh).Nshock;i++) if( fabs((*sh).shock[i][3]-t)<SGL ) if((*sh).shock[i][4]>1){
		((*sh).shock[i][4])--;
		((*sh).shock[i][3])+= (*conf).t_w*(*conf).t_r*(*conf).t_i;
	}
	
	return 1;
}
int _untouch_flight(Aircraft_t **f,int n_f){
	int i;
	for(i=0;i<n_f;i++) (*f)[i].touched = 0;
	return 1;	
}

int _set_angle(CONF_t *c, int try){
	(*c).max_ang = (*c).conf_ang + ((float) try/N_TRY)*( (*c).extr_ang - (*c).conf_ang);
	return 1;
}

int _evolution(Aircraft_t **f,int N_f, CONF_t conf, SHOCK_t sh, TOOL_f tl, long double t ){
	
	//print_time(t);
	
	/*Get the active flight for the current time-step (n_f) */
	int n_f = _get_ready(f,&tl,N_f,t,conf); /*do nothing*/
	//printf("%d Flight Active\n",n_f);
	
	/*shuffle the priority list to solve conflicts*/
	_suffle_list(f,n_f,&tl); 

	/*Select the velocity noise for each flight*/
	_draw_dV(N_f, conf, &tl);
	
	/*Compute the expected position for all the active flight along the look-ahead*/
	_expected_fly(f,n_f,conf,tl); 
	
	/* Evaluate the occupancy for each sector 
	 * (taking in to account also the flight that is going to departures*/
	_evaluate_workload(f,n_f,N_f,tl,conf,t); 
	

#ifdef PLOT 
	//print_workload(tl, conf, "/tmp/work.dat");
	plot_movie(f,n_f,conf,"/tmp/m1.dat");
#endif
	
	/*initialize the touch flag for each flight. If it zero, the flight 
	 * is not touched from the ATC in the current time-step*/
	_untouch_flight(f,n_f);
	
	
	int try=0;
	int f_not_solv=-1;

	_set_angle(&conf,try);
	
	do{	
		/*Evaluate the edgelist for the flight that could interact 
		 * according to the maximum interation distance*/
		_evaluate_neigh(f,n_f,&tl,conf);
		
		/*check conflict for each flight*/
		f_not_solv =_check_safe_events(f,n_f,sh,tl,conf,t); /*put the dv here*/
		
		/*if f_not_solv>0, the f_not_solv flight cannot be solved*/
		if(f_not_solv>=0){
			/*if it does not sol the conflict for 50 trials it exit form the simulation*/
			if(++try> N_TRY) {
				printf("Not Solved, too many trials\n");
				return 0;
			}
			/*Increase the maximum angle of deviation until extr_ang value*/
			_set_angle(&conf,try);
					
			/*It moves the f_not_solv flight on the top of the list,
			 * (no conflict check for the fist of the list*/	 
			_suffle_list_top(f,f_not_solv,&tl);
			
			//return 0;
		}
	}while(f_not_solv>=0);
	
	
#ifdef PLOT 
	plot_movie(f,n_f,conf,"/tmp/m3.dat");
#endif
	/*It updates the shocks to the position of the next time-step*/
	_update_shock(&sh,t,&conf);

	/* It updates the stating position of the aircraft 
	 * to the position of the next time-step*/
	_set_st_point(f, n_f, conf); /*nothing here to do*/
	
	
	return 1;
}

int ABM(Aircraft_t **f, int N_f, CONF_t conf, SHOCK_t sh){

#ifdef PLOT
	/* To plot the movie*/
	FILE *tstream=fopen("/tmp/m1.dat","w");
	fclose(tstream);
	tstream=fopen("/tmp/m3.dat","w");
	fclose(tstream);
	tstream=fopen("/tmp/pp","w");
	fclose(tstream);
	tstream=fopen("/tmp/work.dat","w");
	fclose(tstream);
#endif
	

	
		
	TOOL_f tool;

	_init_tool(&tool,N_f,conf);


	/*Create shocks for the current simulation*/
	_create_shock(&sh,conf);
 
	/*Delay departures of flight*/
 	_delay_on_m1(f,N_f,conf);
 
	/*Modify the velocity of each segment according to a laplacian distribution defined in mABM.h*/
	if(conf.laplacian_vel==1) _laplacian_change_vel(f,N_f);
 
	/*Get the maximum interaction distance to check conflict*/
	_get_d_neigh(&conf,f,N_f);

	/* BEGINNING OF THE SIMULATION */
	int t,step;
	/* definition of the time-step*/
	long double t_stp=(conf.t_r*conf.t_w*conf.t_i);
	
	//printf("Starting Simulation\n");
	
	/*For each time-step it runs evolution*/
	long double NT = ((conf.end_datetime)+t_stp);
	for(t=conf.start_datetime,step=0 ;t<=NT; t+=t_stp, step++ ) if(_evolution(f,N_f, conf,sh,tool,t) == 0) {
		/*If it does not solve the conflict it exit to the main with 0*/		
		_del_tool(&tool,N_f,conf);
		return 0;
	}
	
	
	_del_tool(&tool,N_f,conf);
	
	return 1;
}

