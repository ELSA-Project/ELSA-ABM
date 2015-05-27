/*
 *  mTest.c
 *  ElsaABM_v1
 *
 *  Created by Christian Bongiorno on 13/03/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include "mQuery.h"
#include "mSector.h"
#include "mUtility.h"
#include "mABM.h"
#include "mTest.h"

#include<math.h>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>

void _print_bound( CONF_t c){
	FILE *wstream=fopen("/tmp/bound_latlon.dat","w");
	int i;
	for(i=0;i<c.Nbound;i++) fprintf(wstream, "%Lf\t%Lf\n",c.bound[i][0],c.bound[i][1]);
	fclose(wstream);
	return;
}	
				  

void plot(Aircraft_t f,CONF_t c,char *fr){
	FILE *wstream=fopen(fr,"w");
	int i;
	long double p[2];
	for(i=0;i<f.n_nvp;i++) {
		#ifdef EUCLIDEAN
		fprintf(wstream, "%Lf\t%Lf\n",f.nvp[i][0],f.nvp[i][1]);
		#else
		gall_peter(f.nvp[i], p);
		fprintf(wstream, "%Lf\t%Lf\n",p[0],p[1]);
		#endif
		
	}
	fclose(wstream);
	
	_print_bound(c);
	//system("export DISPLAY=:0.0 && python /Users/profeta/Downloads/Plot.py");
	
	return;
}	
void plot_where(Aircraft_t f,CONF_t c,char *file_w){
	FILE *wstream=fopen(file_w,"w");
	int i;
	for(i=0;i<f.n_nvp;i++) fprintf(wstream, "%Lf\t%Lf\n",f.nvp[i][0],f.nvp[i][1]);
	fclose(wstream);
	
	_print_bound(c);
	//system("export DISPLAY=:0.0 && python /Users/profeta/Downloads/Plot.py");
	
	return;
}	
void print_tmp_point(CONF_t c){
	
	FILE *wstream=fopen("/tmp/pp", "w");
	int i;
	for(i=0;i<c.n_tmp_nvp;i++) fprintf(wstream, "%Lf\t%Lf\n",c.tmp_nvp[i][0],c.tmp_nvp[i][1]);
	fclose(wstream);
	
	_print_bound(c);
	return ;
	
}

void plot_ID(Aircraft_t *flight,int Nflight,CONF_t config,int ID){
	int i;
	for(i=0;i<Nflight;i++) if((flight)[i].ID==ID) break;
	plot(flight[i],config,"/tmp/tt");
	return;
}

void plot_pos(Aircraft_t f,CONF_t c,char *file_w){
	//plot(f,c,"/tmp/pp");
	FILE *wstream=fopen(file_w,"w");
	int i;
	#ifdef EUCLIDEAN
	for(i=0;i<(c.t_w);i++) fprintf(wstream,"%Lf\t%Lf\n",f.pos[i][0],f.pos[i][1]);
	#else
	long double p[2];
	for(i=0;i<(c.t_w);i++){
		gall_peter(f.pos[i], p);
		fprintf(wstream,"%Lf\t%Lf\n",p[0],p[1]);
	}
	#endif
	fclose(wstream);
}

void plot_pos_gall(Aircraft_t f,CONF_t c,char *file_w){
	
	FILE *wstream=fopen(file_w,"w");
	int i;
	long double p[2];
	for(i=0;i<(c.t_w);i++){
		gall_peter(f.pos[i],p);
		fprintf(wstream,"%Lf\t%Lf\n",p[0],p[1]);
	 }
	fclose(wstream);
}

void gall_peter(long double *a,long double *A){
	A[0]=RH*PI*a[1]/(180.*sqrt(2));
	A[1]=RH*sqrt(2)*sin(rad(a[0]));
	return;
}

void plot_bound(CONF_t conf){
	int i;
	long double P[2];
	FILE *wstream=fopen("DATA/bound_proj.dat", "w");
	for(i=0;i<conf.Nbound;i++){
		gall_peter(conf.bound[i],P);
		fprintf(wstream, "%Lf\t%Lf\n",P[0],P[1]);
	}
	fclose(wstream);
	return;
}

void plot_movie( Aircraft_t **f,int N_f,CONF_t conf,char *fw ){
	FILE *postream=fopen(fw, "a");

	int i,j;
	long double P[2];
	for(j=0;j<(conf.t_w*conf.t_r);j++) {
		for(i=0;i<N_f;i++) if((*f)[i].pos[j][3]==1. && (*f)[i].pos[j][0]!=SAFE)
		{
			//gall_peter((*f)[i].pos[j],P);
			//fprintf(postream,"%Lf\t%Lf\t%Lf\t",P[0],P[1],(*f)[i].pos[j][2]);
			fprintf(postream,"%Lf\t%Lf\t%Lf\t",(*f)[i].pos[j][0],(*f)[i].pos[j][1],(*f)[i].pos[j][2]);
			
			fprintf(postream,"%d\t",(*f)[i].ID);
		}
		fprintf(postream,"\n");
	}
	fclose(postream);
	return;
}	
void print_nvp(Aircraft_t f){
	int i,h;
	for(i=0;i<f.n_nvp;i++) {
		printf("%d]\t",i);
		for(h=0;h<DPOS;h++) printf("%Lf\t",f.nvp[i][h]);
		if(i==f.bound[0]||i==f.bound[1]) printf("1");
		printf("\n");
	}
	return;
}



void cheak_inside_pos(Aircraft_t *f,int n_f,CONF_t conf){
	int i,j;
	for(i=0;i<n_f;i++) for(j=0;j<conf.t_w;j++) if(f[i].pos[j][3]==0 &&point_in_polygon(f[i].pos[j], conf.bound, conf.Nbound)==1){
		printf("SUSU\n");
	}
	return;
}


int cheak_nan_pos(Aircraft_t *f,CONF_t conf){
	int i;
	for(i=0;i<conf.t_w;i++) if(isnan((*f).pos[i][0])) return 1;
	return 0;
	
}

void print_time(long double t){
	struct tm *T;
	char buff[100];
	time_t pT = (time_t) t;
	
	T = localtime(&pT);
				
	strftime(buff,100,"%Y-%m-%d %H:%M:%S",T);
	printf("%s\n",buff);
	
	return;
	
}


