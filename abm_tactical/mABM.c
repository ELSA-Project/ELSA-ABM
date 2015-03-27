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
	
	fprintf(wstream,"%d\tNflight\n",Nflight);
	int i,j,h,T[DPOS];
	for(h=0;h<Nflight;h++){
		for(i=0;i<Nflight;i++) if(Flight[h].ID==flight[i].ID) { 
			fprintf(wstream,"%d\t%d\t",flight[i].ID,flight[i].n_nvp);
			for(j=0;j<flight[i].n_nvp;j++){
				time_to_int(flight[i].time[j],T);
				fprintf(wstream,"%Lf,%Lf,%.0Lf,2010-05-04 %d:%d:%d:%d\t",flight[i].nvp[j][0],flight[i].nvp[j][1],flight[i].nvp[j][2],T[0],T[1],T[2],T[3]);
			}
			fprintf(wstream, "\n");
		}
	}
	//fprintf(wstream, "---\n");
	fclose(wstream);
	
	return;
}

int del_flight(Aircraft_t **f, int N,Aircraft_t *F){
	int i,j;
	for(i=0;i<N;i++){
		for(j=0;j<N;j++) if((F)[j].ID==(*f)[i].ID) break;
		ffree_2D((*f)[i].nvp, (F)[j].n_nvp);
		//ffree_2D((*f)[i].nvp, (*f)[i].n_nvp);
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
	int i,j;
	
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
	
	#ifdef CAPACITY
	free((*t).workload);
	#endif
	
	return 1;
}

int _create_shock(SHOCK_t *sh,CONF_t conf){
	
	int i,coin, niter= (int)(DAY/ (conf.t_r*conf.t_w*conf.t_i));
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
		*l_nvp=haversine_distance(nvp[*j],nvp[*j+1]);
		
		return _nvp_to_close(d,t_c,vel,dV,nvp,l_nvp,j,n_nvp,time,t);
	}
	else return 1;
}


/*Calculate the Position Array in the time-step . Return Zero if the Aircraft will land in the current timestep.*/
int _position(Aircraft_t *f,long double *st_point, int t_wind, long double time_step, long double t_r, long double dV){
	int i,j;
	long double l_nvp,d,t_c;
	
	long double t0 = (*f).time[(*f).st_indx-1] + haversine_distance(st_point, (*f).nvp[(*f).st_indx-1])/(*f).vel[(*f).st_indx-1];
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
	
	l_nvp=haversine_distance(st_point,nvp[st_indx]);

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
			
			l_nvp=haversine_distance(nvp[j],nvp[j+1]);
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
			coord(vel[j], nvp[j], nvp[j+1], pos[i+1], t_c);
			pos[i+1][2]=nvp[j][2];
			pos[i+1][3]=nvp[j][3];
			pos[i+1][4]=nvp[j][4];
			
		}
		else {
		    coord(vel[j], pos[i], nvp[j+1], pos[i+1], time_step);
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
			
			l_nvp=haversine_distance(nvp[j],nvp[j+1]);
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
			coord(vel[j]*(1. + dV), nvp[j], nvp[j+1], pos[i+1], t_c);
			pos[i+1][2]=nvp[j][2];
			pos[i+1][3]=nvp[j][3];
			pos[i+1][4]=nvp[j][4];
			
		}
		else {
		    coord(vel[j]*(1. + dV), pos[i], nvp[j+1], pos[i+1], time_step);
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
int _sort_ready(Aircraft_t **f,int N_f){
	int i,j;
	Aircraft_t comodo;
	
	for(i=0;i<N_f;i++){
		if((*f)[i].ready) continue;
		else for(j=i+1;j<N_f;j++) if((*f)[j].ready){
			comodo=(*f)[i];
			(*f)[i]=(*f)[j];
			(*f)[j]=comodo;
		}
	}
	return 1;
}

int _set_st_point(Aircraft_t **f,int N_f,CONF_t conf){
	
	int i,old_st,mynull;
	/*The real position for departures*/
	int t_x = (conf.t_w*conf.t_r);
	for(i=0;i<N_f;i++)if((*f)[i].ready){
		if((*f)[i].pos[t_x-1][0]==SAFE){
			(*f)[i].ready=0;
			continue;
		}
		(*f)[i].st_point[0]=(*f)[i].pos[t_x-1][0];
		(*f)[i].st_point[1]=(*f)[i].pos[t_x-1][1];
		(*f)[i].st_point[2]=(*f)[i].pos[t_x-1][2];
		(*f)[i].st_point[3]=(*f)[i].pos[t_x-1][3];
		(*f)[i].st_point[4]=(*f)[i].pos[t_x-1][3];
		
		old_st=(*f)[i].st_indx;
		if((*f)[i].st_point[0]==SAFE) (*f)[i].st_indx=(*f)[i].n_nvp;
		else (*f)[i].st_indx = find_st_indx((*f)[i].nvp, (*f)[i].n_nvp, (*f)[i].st_point,(*f)[i].st_indx,(*f)[i].pos[0]);

		
		if((*f)[i].ready!=0&&(*f)[i].st_indx==-1){
			int j;
			for(j=0;j<((*f)[i].n_nvp-1);j++) printf("%Lf ",(*f)[i].vel[j]);
			printf("\n");
			plot_pos((*f)[i], conf);
			plot((*f)[i],conf,"/tmp/tt");
			plot_where((*f)[i],conf,"/tmp/nvp_f");
			printf("BUBU %d ID\n",(*f)[i].ID);
			(*f)[i].st_indx=old_st;
			printf("%d\t%d\n",old_st,(*f)[i].n_nvp);
			//scanf("%d",&mynull);
			//BuG("In HERE\n");
			//exit(0);
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
			if(haversine_distance((*f)[i].st_point,(*f)[j].st_point)<conf.d_neigh){
				(*tl).neigh[j][(*tl).n_neigh[j]]=i;
				((*tl).n_neigh[j])+=1;
			}
		}
	}
		
	return 1;
	
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
		plot_pos((*f),conf);
		plot_where((*f),conf,"/tmp/nvp_f");
	//if(1){
		printf("st: %Lf\t%Lf\n",(*f).st_point[0],(*f).st_point[1]);
		printf("%Lf\n",distance_point_segment((*f).nvp[1], (*f).nvp[2], (*f).pos[conf.t_w-1]));
		
		printf("%d st_indx\n",(*f).st_indx);
		int i;
		for(i=0;i<conf.t_w;i++) printf("%Lf\t%Lf\n",(*f).pos[i][0],(*f).pos[i][1]);
		plot_pos((*f), conf);
		printf("ID: %d\n",(*f).ID);
		BuG("Error on st_indx on Departures\n");
	}
#endif
	
	return 1;
}

/* Put to one ready flag for plane on departures and sort in the first position them 
 and calculate starting point */
int _get_ready(Aircraft_t **f, int N_f,long double t,CONF_t conf){
	int i,j;
	long double t_stp=(conf.t_r*conf.t_w*conf.t_i);

	for(i=0;i<N_f;i++) {
		if((*f)[i].ready==0) {
			if( (*f)[i].time[0]>=t&& (*f)[i].time[0]< (t+t_stp) )  {
				(*f)[i].ready=1;
				(*f)[i].tp = ((*f)[i].time[0]-t)/conf.t_i;
				(*f)[i].st_indx = 1;
				for(j=0;j<DPOS;j++) (*f)[i].st_point[j] = (*f)[i].nvp[0][j];
			}
		}
		else (*f)[i].tp = 0;
	}
	
	int n_f=0;
	for(i=0;i<N_f;i++) if((*f)[i].ready) n_f++;
	
	_sort_ready(f,N_f);
	
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
	Aircraft_t comodo;
	//Aircraft_t xfli=(*f)[n_f];
	
	for(i=n_f;i>1;i--){
		comodo = (*f)[i];
		(*f)[i] = (*f)[i-1];
		(*f)[i-1]=comodo;
		
	}
	
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
	
	for(indx=1;indx<conf.t_w;indx++) if (d[indx]<conf.d_thr) break;
	if(indx==conf.t_w) return 0;

	/*return navpoint immediatly after the collision*/
	if(isbetween((*f).st_point, (*f).nvp[(*f).st_indx], (*f).pos[indx])) return (*f).st_indx;
	for(i=(*f).st_indx ; i<((*f).n_nvp-1);i++){
		if(isbetween((*f).nvp[i],(*f).nvp[i+1],(*f).pos[indx])) {
		    return i+1;}
	}
	
	return -1;
}

int _minimum_flight_distance(long double **pos,Aircraft_t **f,int N_f,int N,TOOL_f tl){
	int h,i,j;
	long double min;
	
	for(i=0;i<N;i++){
		if((int)pos[i][3]&&N_f>0) {
			if(((int)pos[i][2])!=((int)(*f)[0].pos[i][2])||((int) (*f)[0].pos[i][3])==0) min=SAFE;
			else  min=haversine_distance(pos[i],(*f)[0].pos[i]);
			
			
			for(h=0;h<tl.n_neigh[N_f];h++){
				j=tl.neigh[N_f][h];
				
				if( fabsl((int)pos[i][2]-((int)(*f)[j].pos[i][2]))<9.99&&(int) (*f)[j].pos[i][3]==1){
					tl.dist[i]=haversine_distance(pos[i],(*f)[j].pos[i]);
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
	return haversine_distance(p,shock)<shock[2];
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
int _check_risk(long double *d,CONF_t conf,int t_w){
	int i,risk;
	for(i=1,risk=0;i<t_w;i++) if(d[i]!=SAFE) if(d[i]<=conf.d_thr) risk=1;
	
	return risk;
}


int _get_d_neigh(CONF_t *conf,Aircraft_t **f,int N_f){
	int i,j;
	long double max_v=0;
	for(i=0;i<N_f;i++) for(j=0;j<((*f)[i].n_nvp-1);j++) if((*f)[i].vel[j]>max_v) max_v = (*f)[i].vel[j];
	
	(*conf).d_neigh = 2.*(2.*((*conf).t_w*(*conf).t_i)*max_v*(1+(*conf).sig_V)) + (*conf).d_thr ;
	
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
	/*if(safe==1) for(i=0;i<conf.n_tmp_nvp;i++) {
		(*tl).temp_angle[i]=0;
		return 1;
	}*/
	for(i=0;i<conf.n_tmp_nvp;i++) if(haversine_distance((*f).st_point,conf.tmp_nvp[i])<DTMP_P){
		if( deg(PI-angle_direction((*f).st_point,conf.tmp_nvp[i] , (*f).nvp[safe+1])) >(90.)) (*tl).temp_angle[i]=angle_direction((*f).nvp[safe-1], (*f).nvp[safe], conf.tmp_nvp[i]);
		 
		else (*tl).temp_angle[i]=PI;
	}
	
	return 1;
}

int _select_candidate_tmp_nvp_shortpath(Aircraft_t *f,CONF_t conf, TOOL_f *tl, int safe){
	int i;
	
	_temp_angle_nvp(f,conf,tl,(*f).st_indx);

	for(i=0,(*tl).n_sel_nvp=0 ;i<conf.n_tmp_nvp;i++) if(haversine_distance((*f).st_point,conf.tmp_nvp[i])<DTMP_P) if((*tl).temp_angle[i]<conf.max_ang) if(frand(0,100)<50){
		(*tl).sel_nvp_index[(*tl).n_sel_nvp][0]= haversine_distance((*f).st_point, (conf).tmp_nvp[i])+haversine_distance( (conf).tmp_nvp[i],(*f).nvp[safe+1]);
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

	if(conf.shortest_path) _select_candidate_tmp_nvp_shortpath(f,conf,&tl,(*f).st_indx);
	else 	_select_candidate_tmp_nvp(f,conf,&tl,(*f).st_indx);
	
	int i,j;
	int n_old=unsafe - (*f).st_indx + 1;

	/*Velocity*/
	long double *old_vel=falloc_vec((*f).n_nvp-1);
	for(i=0;i<((*f).n_nvp-1);i++) old_vel[i]=(*f).vel[i];
	
	//Sarebbe nold-- error
	int njump=unsafe-(*f).st_indx+2;
	
	long double vm=mean(&((*f).vel[(*f).st_indx-1]),unsafe-(*f).st_indx+2);
	(*f).vel[(*f).st_indx-1]=vm;
	(*f).vel[(*f).st_indx]=vm;
	
	for(i=(*f).st_indx+1;i<((*f).n_nvp-2);i++) (*f).vel[i]=(*f).vel[i+1];

	/*Navigation Point*/
	long double **old_nvp=falloc_matrix((*f).n_nvp,DPOS);
	int olf=(*f).n_nvp;
	for(i=0;i<(*f).n_nvp;i++) for(j=0;j<DPOS;j++) old_nvp[i][j]=(*f).nvp[i][j];
		
	int rp_temp;
	
	for(i=unsafe+1,rp_temp=0;i<(*f).n_nvp;i++,rp_temp++) {
		for(j=0;j<DPOS;j++) (*f).nvp[i-n_old+1][j]=(*f).nvp[i][j];
	}

	(*f).n_nvp=(*f).n_nvp -  n_old+1;
	
	int solved,h,dj,l,m;
	//double newv;
	long double temp_rout;

	/*For every comeback point after collision*/
	int longest_rer =  _calculate_longest_direct(f,tl,conf,1);
	for(dj=0,j=((*f).st_indx+1);j<longest_rer;j++,dj++){
		
		
		//if(dj>0) newv=((*f).vel[(*f).st_indx]*njump+(*f).vel[(*f).st_indx+1])/(++njump);
		
		/*Modify trajectory reducing point*/
		for(h=((*f).st_indx+1),m=0;h<((*f).n_nvp-dj);h++,m++) {
			for(l=0;l<DPOS;l++) (*f).nvp[h][l]=(*f).nvp[(*f).st_indx+dj+m+1][l];
			
			(*f).vel[h-1]=(*f).vel[(*f).st_indx+dj+m];
		}

		
		(*f).n_nvp=(*f).n_nvp-dj;
			
		for(i=0;i<tl.n_sel_nvp;i++){
			
			
			/*lat and lon*/
			(*f).nvp[(*f).st_indx][0]=conf.tmp_nvp[(int) tl.sel_nvp_index[i][1]][0];
			(*f).nvp[(*f).st_indx][1]=conf.tmp_nvp[(int) tl.sel_nvp_index[i][1]][1];
			(*f).nvp[(*f).st_indx][2]=old_nvp[unsafe][2];
			(*f).nvp[(*f).st_indx][3]=old_nvp[unsafe][3];

			temp_rout = haversine_distance((*f).st_point, (*f).nvp[(*f).st_indx])/(*f).vel[(*f).st_indx-1]+haversine_distance((*f).nvp[(*f).st_indx+1], (*f).nvp[(*f).st_indx])/(*f).vel[(*f).st_indx];
			if(temp_rout>(conf.t_w*2*conf.t_i)) {
				//printf("Too dist %Lf\n",temp_rout/60.);
				continue;
			}
			/*if the comeback angle is less then max_ang*/
			if(angle_direction((*f).nvp[(*f).st_indx], (*f).nvp[j], (*f).nvp[j+1])>conf.max_ang) continue;

			_position(f , (*f).st_point , conf.t_w, conf.t_i, conf.t_r, dV);
			_minimum_flight_distance((*f).pos,&flight,N_f,conf.t_w,tl);
			
			_checkShockareaRoute((*f).pos,conf.t_w, sh,tl.dist,t);
			solved=_check_risk(tl.dist,conf,conf.t_w);
		
			/*if solved*/
			if(solved==0) {
				//DA CONTROLLARE
				//printf("dj=%d\t IDnew %d\tunsafe %d, Angl %Lf\n",dj,(*f).ID,unsafe,angle_direction((*f).nvp[(*f).st_indx], (*f).nvp[j], (*f).nvp[j+1]));
				_position(f , (*f).st_point , conf.t_w*2, conf.t_i, conf.t_r, dV);
				ffree_2D(old_nvp, olf);
				free(old_vel);
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
	safe=_check_risk(tl.dist,conf,conf.t_w);
	return safe;
}

int _is_close_nvp(Aircraft_t f){
	
	if(haversine_distance ( (f).nvp[(f).st_indx-1],(f).st_point)<500) return 0;
	else return 1;
}

int _find_end_point(Aircraft_t *f,CONF_t conf){
	
	int i;
	for(i=conf.t_w-1;i>0;i--) if((*f).pos[i][0]!=SAFE) break;
	int end = find_p_indx((*f).nvp,(*f).n_nvp,(*f).pos[i]);
	if (end==0) {
		return (*f).st_indx;
		//return (*f).n_nvp;
	}
	else return end;	
}

int _change_flvl(Aircraft_t *f,Aircraft_t **flight,int N_f,CONF_t conf,TOOL_f tl,SHOCK_t sh,long double t){
	int unsafe = (*f).st_indx-1;
	
	int endp = _find_end_point(f,conf);
	
	long double *h=falloc_vec(endp-((*f).st_indx-1));
	int i,j;
	for(i=(*f).st_indx-1,j=0;i<endp;i++,j++) h[j]=(*f).nvp[i][2];
	
	long double f_lvl_on=(*f).nvp[  unsafe ][2];

	if(!_try_flvl(f,flight,N_f,conf,tl,sh,f_lvl_on+20.,unsafe,endp,t)){
		free(h);
		return 0;
	 }
	if(!_try_flvl(f,flight,N_f,conf,tl,sh,f_lvl_on-20.,unsafe,endp,t)) {
		free(h);
		return 0;
	}

	for(i=(*f).st_indx-1,j=0;i<endp;i++,j++) (*f).nvp[i][2]=h[j];
	free(h);
	
	return 1;
}

double _calculate_optimum_direct(Aircraft_t *f,int on){

	double d1=haversine_distance((*f).st_point,(*f).nvp[(*f).st_indx+on]);
	double d2=haversine_distance((*f).st_point,(*f).nvp[(*f).st_indx]);
	int i;
	for(i=0;i<on;i++){
		d2+=haversine_distance((*f).nvp[(*f).st_indx+i],(*f).nvp[(*f).st_indx+i+1]);
	}

	return d2-d1;							 
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
			//~ if (rer) printf("rer Overfull Capacity %d \t %d -> %d\t%d\n",(conf.capacy[sect]),tl.workload[sect],st_in,i-1);
			//~ else  printf("dir Overfull Capacity %d \t %d -> %d\t%d\n",(conf.capacy[sect]),tl.workload[sect],st_in,i-1);
			
			//return (*f).n_nvp-plus;

			return i-1;
		}
	}

	return (*f).n_nvp-plus;
}	
			
int _direct(Aircraft_t *f,Aircraft_t *flight,int N_f,CONF_t conf, TOOL_f tl,SHOCK_t sh,long double tt){
	/*If the new path is less than the original*/
	long double diff[]={0,0};
	int old_st_indx=(*f).st_indx;
	int i=(*f).st_indx;
	
	int o;
	
	int j;
			
	long double t=haversine_distance( (*f).pos[0], (*f).nvp[(*f).st_indx+1] )/(*f).vel[i];
	
	diff[0]=_calculate_optimum_direct(f, 1);
	if(diff[0]<1000.) return 0;
		
	int h;
	int longest_direct = _calculate_longest_direct(f,tl,conf,0);
	for(i=((*f).st_indx+1),h=1;i<(longest_direct) &&t<1200.;i++,h++) {
		diff[1]=_calculate_optimum_direct(f, h+1);
		if ((diff[1]-diff[0])<1000.){
			//printf("diff %Lf\n",diff[1]-diff[0]);
			i--;
			break;
		}
		
		diff[0]=diff[1];
		t=haversine_distance((*f).pos[0], (*f).nvp[i+1])/(*f).vel[i];

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

	_position(f, (*f).st_point, conf.t_w*2, conf.t_i, conf.t_r, 0.); //TODO: maybe we should change this.
	
	_minimum_flight_distance((*f).pos,&flight,N_f,2*conf.t_w,tl);
	
	_checkShockareaRoute((*f).pos,conf.t_w*2, sh,tl.dist,tt);
	if(_check_risk(tl.dist,conf,conf.t_w*2)){
		for(o=0;o<(old_n_nvp-1);o++) {
			for(j=0;j<DPOS;j++) (*f).nvp[o][j]=old_nvp[o][j];
			(*f).vel[o]=old_vel[o];
		}
		for(j=0;j<DPOS;j++) (*f).nvp[o][j]=old_nvp[o][j];

		_position(f, (*f).st_point, conf.t_w*2, conf.t_i, conf.t_r, 0.);
		ffree_2D(old_nvp,old_n_nvp);
		free(old_vel);

		return 0;
	}
	else{
		
		_position(f, (*f).st_point, conf.t_w*2, conf.t_i, conf.t_r, 0.);

	}
		
	ffree_2D(old_nvp,old_n_nvp);
	free(old_vel);
	
	return 1;
}

int _check_safe_events(Aircraft_t **f, int N_f, SHOCK_t sh,TOOL_f tl, CONF_t conf, long double t){
	
	int i,unsafe;
	int j;

	for(i=1;i<N_f;i++) if((*f)[i].ready){
		
		_minimum_flight_distance((*f)[i].pos,f,i,conf.t_w,tl);
		
		_checkShockareaRoute((*f)[i].pos,conf.t_w, sh,tl.dist,t);
		unsafe=_check_risk(tl.dist,conf,conf.t_w);
		
		if(unsafe) {
			unsafe=_checkFlightsCollision(tl.dist, conf, &(*f)[i]); 

			unsafe=_reroute(&((*f)[i]),(*f),i,sh,conf,tl,unsafe,t);
			
			if(unsafe) 
				unsafe=_change_flvl(&(*f)[i],f,i,conf,tl,sh,t);
	
			if(unsafe) return i;
			
		}
		else if(frand(0,1)<conf.direct_thr) if((*f)[i].st_indx< ((*f)[i].n_nvp-2)&&(*f)[i].st_indx>1) {
			//add_nvp_st_pt(&(*f)[i]);
			_direct(&(*f)[i],(*f),i,conf,tl,sh,t);
		}
		 
	}
	
	
	return -1;
}


/* Compute who is ready*/
int _expected_fly(Aircraft_t **f, int N_f, CONF_t conf, TOOL_f tl){
	int i;
	for(i=0;i<N_f;i++){
		//printf("%d: %d\n", i, (*f)[i].ready);
		if((*f)[i].ready){
			//printf("%d: %d\n", i, (*f)[i].ready);
			if(!_position(&(*f)[i], (*f)[i].st_point, conf.t_w*2, conf.t_i, conf.t_r, tl.dV[i]) && (*f)[i].pos[0][0]==SAFE) {
				(*f)[i].ready=0;
			}/*put the dv[i] here*/
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

int _evaluate_workload(Aircraft_t **f,int n_f,TOOL_f tl,CONF_t conf){
	int i,j,t;
	
	//~ for(j=0;j<(2*conf.t_w);j++){
		//~ for(i=1;i<=conf.n_sect;i++) tl.workload[j][i]=0;
		//~ 
		//~ for(i=0;i<n_f;i++) {
			//~ if( (int) (*f)[i].pos[j][4]  > conf.n_sect  || (int) (*f)[i].pos[j][4] <0 ){
				 //~ printf(" %d su %d\n", (int) (*f)[i].pos[j][4] ,conf.n_sect);
				 //~ exit(0);
			 //~ }
			//~ (tl.workload[j][ (int) (*f)[i].pos[j][4] ])++;
			//~ 
		//~ }
	//~ }
	
	//~ for(i=1;i<conf.n_sect;i++) {
		//~ tl.workload[i]=0;
		//~ for(j=0;j<n_f;j++) for(t=0;t<(2*conf.t_w);t++) if( (int)(*f)[j].pos[t][4] == i ) {
			//~ (tl.workload[i])++;
			//~ break;
		//~ }
	//~ }
	int walk[(conf).n_sect];
	
	for(i=0;i<conf.n_sect;i++) tl.workload[i]=0;
	for(i=0 ;i<n_f;i++) {
		
		for(j=0;j<(conf).n_sect;j++) walk[j]=0;
		
		t = haversine_distance((*f)[i].st_point,(*f)[i].nvp[(*f)[i].st_indx])/(*f)[i].vel[(*f)[i].st_indx-1];
		( walk[ (int) (*f)[i].pos[1][4]] )++;
		
		for(j=(*f)[i].st_indx;j<((*f)[i].n_nvp-1);j++) {
			t += haversine_distance((*f)[i].nvp[j],(*f)[i].nvp[j+1])/(*f)[i].vel[j];
			if( t > 3600. ) break;
			(walk[(int) (*f)[i].nvp[j][4]])++;
		}
		for(j=1;j<(conf).n_sect;j++) if(walk[j]!=0) (tl.workload[j])++;
		
		
	}
	
	
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

int _evolution(Aircraft_t **f,int N_f, CONF_t conf, SHOCK_t sh, TOOL_f tl, long double t ){
		
	//printf("%.3Lf\n",t/3600.); 	/*Print time*/
	int n_f = _get_ready(f,N_f,t,conf); /*do nothing*/

	_suffle_list(f,n_f,&tl); /*do nothing (??) */

	_draw_dV(N_f, conf, &tl);
	//printf("Coin2\n");
	
	
	_expected_fly(f,n_f,conf,tl); /*put dv here*/
	//printf("Coin3\n");
	
#ifdef CAPACITY	
	_evaluate_workload(f,n_f,tl,conf); //DA CONTROLLARE
	//print_workload(tl, conf, "/tmp/work.dat");
#endif

#ifdef PLOT 
	//print_workload(tl, conf, "/tmp/work.dat");
	plot_movie(f,n_f,conf,"/tmp/m1.dat");
#endif
	
	int try=0;
	int f_not_solv=-1;

	do{	
		_evaluate_neigh(f,n_f,&tl,conf);
		
		f_not_solv =_check_safe_events(f,n_f,sh,tl,conf,t); /*put the dv here*/
		
		if(f_not_solv>=0){
		
			if(++try> 500) {
				printf("Not Solved, too many trials\n");
				return 0;
			}				 
			_suffle_list_top(f,f_not_solv,&tl);	
		}
	}while(f_not_solv>=0);
	
	/*
	int i,r;
	//printf("Nflight %d\n",n_f);
	for(i=1;i<n_f;i++) {
		_minimum_flight_distance( (*f)[i].pos,f,i-1,conf.t_w,tl);
		
		if(_check_risk(tl.dist,conf,conf.t_w)){
			for(r=0;r<conf.t_w;r++) printf("%Lf\n",tl.dist[r]);
			printf("Try%d\n",try);
			 BuG("Not Solved Why??\n");
		 }
	
	}
	*/
	
#ifdef PLOT 
	plot_movie(f,n_f,conf,"/tmp/m3.dat");
#endif
	
	_update_shock(&sh,t,&conf);

	_set_st_point(f, n_f, conf); /*nothing here to do*/

	
	
	return 1;
}

int ABM(Aircraft_t **f, int N_f, CONF_t conf, SHOCK_t sh){

#ifdef PLOT
	FILE *tstream=fopen("/tmp/m1.dat","w");
	fclose(tstream);
	tstream=fopen("/tmp/m3.dat","w");
	fclose(tstream);
	tstream=fopen("/tmp/pp","w");
	fclose(tstream);
	tstream=fopen("/tmp/work.dat","w");
	fclose(tstream);
#endif
	

	// plot((*f)[i], conf);
	TOOL_f tool;
	//printf("Pouic1\n");
	_init_tool(&tool,N_f,conf);

	_create_shock(&sh,conf);
 
 	_delay_on_m1(f,N_f,conf);
 
	if(conf.laplacian_vel==1) _laplacian_change_vel(f,N_f);
 
	_get_d_neigh(&conf,f,N_f);

	//printf("Inizio Evoluzione\n");
	int t,step;
	long double t_stp=(conf.t_r*conf.t_w*conf.t_i);

	for(t=t_stp,step=0 ;t<DAY; t+=t_stp, step++ ) if(_evolution(f,N_f, conf,sh,tool,t) == 0) {		
		_del_tool(&tool,N_f,conf);
		return 0;
	}
	
	//printf("Pouic7\n");
	
	_del_tool(&tool,N_f,conf);
	
	return 1;
}

