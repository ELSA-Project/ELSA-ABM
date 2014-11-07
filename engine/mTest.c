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
	for(i=0;i<f.n_nvp;i++) fprintf(wstream, "%Lf\t%Lf\n",f.nvp[i][0],f.nvp[i][1]);
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

void plot_pos(Aircraft_t f,CONF_t c){
	plot(f,c,"/tmp/pp");
	FILE *wstream=fopen("/tmp/pp","w");
	int i;
	for(i=0;i<c.t_w;i++) fprintf(wstream,"%Lf\t%Lf\n",f.pos[i][0],f.pos[i][1]);
	fclose(wstream);
}
void _gall_peter(long double *a,long double *A){
	A[0]=RH*PI*a[1]/(180.*sqrt(2));
	A[1]=RH*sqrt(2)*sin(rad(a[0]));
	return;
}

void plot_bound(CONF_t conf){
	int i;
	long double P[2];
	FILE *wstream=fopen("DATA/bound_proj.dat", "w");
	for(i=0;i<conf.Nbound;i++){
		_gall_peter(conf.bound[i],P);
		fprintf(wstream, "%Lf\t%Lf\n",P[0],P[1]);
	}
	fclose(wstream);
	return;
}

void plot_movie( Aircraft_t **f,int N_f,CONF_t conf){
	FILE *postream=fopen("DATA/pos.dat", "a");

	int i,j;
	long double P[2];
	for(j=0;j<(conf.t_r*conf.t_w -2);j++) {
		for(i=0;i<N_f;i++) if((*f)[i].pos[j][3]==1. && (*f)[i].pos[j][0]!=SAFE)
		{
			_gall_peter((*f)[i].pos[j],P);
			fprintf(postream,"%Lf\t%Lf\t%Lf\t",P[0],P[1],(*f)[i].pos[j][2]);
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
		for(h=0;h<4;h++) printf("%Lf\t",f.nvp[i][h]);
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


