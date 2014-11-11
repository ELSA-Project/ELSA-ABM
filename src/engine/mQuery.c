/*
 *  mQuery.c
 *  ElsaABM
 *
 *  Created by Christian Bongiorno on 06/03/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "mQuery.h"
#include "mUtility.h"
#include "mSector.h" 

#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>

/*Take string as HH:MM:SS and return the number of seconds*/
long double _convert_time(char *c ){
	long double time;
	int H,m,s, i,f;
	
	H=atoi(c);
	for(i=0;c[i]!=':';i++);
	m=atoi(&c[++i]);
	for(;c[i]!=':';i++);
	s=atoi(&c[++i]);
	for(;c[i]!=':';i++);
	f=atoi(&c[++i]);
	
	time= (H*3600.)+(m*60.) + (long double) s + (long double) f/1000000.;
	
	return time;
}

int _calculate_velocity(Aircraft_t *flight,int Nflight){
	int i,j;
	long double t;
	for(i=0;i<Nflight;i++){
		for(j=0;j<(flight[i].n_nvp-1);j++) {
			t=(flight[i].time[j+1]-flight[i].time[j]);
			if(t>0.001) flight[i].vel[j]=haversine_distance(flight[i].nvp[j], flight[i].nvp[j+1])/t;
			else  flight[i].vel[j]=haversine_distance(flight[i].nvp[j], flight[i].nvp[j+1])/1.;
		}	
	}
	return 1;
}

int get_M1(char *m1_file,Aircraft_t **flight){
	
	FILE *rstream;
	
	rstream=fopen(m1_file,"r");
	if(rstream==NULL) BuG("M1 File doesn't exist\n");
	
	char c[R_BUFF];
	fgets(c, R_BUFF, rstream);
	
	int Nflight=atoi(c);
	//Nflight=600;
	int i,j,h;

	(*flight)=( Aircraft_t* ) malloc(Nflight*sizeof(Aircraft_t));
	
	for(i=0;i<Nflight;i++){
		if(!fgets(c, R_BUFF, rstream)) BuG("BUG in M1 File -lx0\n");
		
		(*flight)[i].ID=atoi(c);
		for(j=1;c[j]!='\t';j++);
		
		(*flight)[i].n_nvp=atoi(&c[++j]);
		
		(*flight)[i].nvp=falloc_matrix((*flight)[i].n_nvp,4);
		(*flight)[i].time=falloc_vec((*flight)[i].n_nvp);
		(*flight)[i].vel=falloc_vec((*flight)[i].n_nvp - 1);

		for(h=0;h<(*flight)[i].n_nvp;h++){
			for(++j;c[j]!='\t'&&c[j]!='\0'&&c[j]!='\n';j++);
			if(c[j]=='\0'||c[j]=='\n') BuG("BUG in M1 File -lx1\n");
			(*flight)[i].nvp[h][0]=atof(&c[++j]);
			
			for(++j;c[j]!=','&&c[j]!='\0'&&c[j]!='\n';j++);
			if(c[j]=='\0') BuG("BUG in M1 File -lx2\n");
			(*flight)[i].nvp[h][1]=atof(&c[++j]);
			
			for(++j;c[j]!=','&&c[j]!='\0'&&c[j]!='\n';j++);
			if(c[j]=='\0') BuG("BUG in M1 File -lx3\n");
			(*flight)[i].nvp[h][2]=atof(&c[++j]);
			
			
			for(++j;c[j]!=','&&c[j]!='\0';j++);
			if(c[j]=='\0') BuG("BUG in M1 File -lx4\n");
			for(++j;c[j]!=' '&&c[j]!='\0';j++);
			if(c[j]=='\0') BuG("BUG in M1 File -lx5\n");
			(*flight)[i].time[h]=_convert_time(&c[++j]);
		}
	}
		
	_calculate_velocity((*flight),Nflight);
	
	/*for(i=0;i<Nflight;i++) if((*flight)[i].n_nvp<=3){
		remove_aircraft(flight, &Nflight, i--);
		Nflight--;
	}*/
	
	return Nflight;
}

long double _find_value_string(char *config_file,char *label){
	
	FILE *rstream=fopen(config_file, "r");
	if(rstream==NULL) BuG("BUG - configuration file doesn't exist\n");
	
	char c[R_BUFF];
	int i,lsize;
	
	for(lsize=0;label[lsize]!='\0';lsize++);
	
	while (fgets(c, R_BUFF, rstream)) {
		if(c[0]=='#'||c[0]=='\n'||c[0]==' ') continue;
		for(i=0;c[i]!='#'&&c[i]!='\0';i++);
		if(c[i]=='\0') BuG("configuration file not standard\n");
		if(!memcmp(&c[++i], label, lsize)) {
			fclose(rstream);
			return atof(c);
		}
	}
	
	fclose(rstream);
	printf("Impossible to fine %s in config-file\n",label);
	exit(0);
}

int get_configuration(char *config_file,CONF_t *config){
	
	(*config).max_ang = _find_value_string(config_file,"max_ang");
	(*config).nsim = (int) _find_value_string(config_file,"nsim");
	(*config).direct_thr = _find_value_string(config_file,"direct_thr");
	(*config).xdelay = _find_value_string(config_file,"xdelay");
	(*config).pdelay = _find_value_string(config_file,"pdelay");
	(*config).laplacian_vel = (int) _find_value_string(config_file,"laplacian_vel");
	(*config).Nm_shock = (int) _find_value_string(config_file,"Nm_shock");
	(*config).radius = _find_value_string(config_file,"radius");
	(*config).t_w = (int) _find_value_string(config_file,"t_w");
	(*config).t_i = _find_value_string(config_file,"t_i");
	(*config).t_r = _find_value_string(config_file,"t_r");
	(*config).shortest_path = (int) _find_value_string(config_file,"shortest_path");
	(*config).d_thr = (int) _find_value_string(config_file,"d_thr");
	(*config).f_lvl[0] = _find_value_string(config_file, "shock_f_lvl_min");
	(*config).f_lvl[1] = _find_value_string(config_file, "shock_f_lvl_max");
	(*config).geom = _find_value_string(config_file, "geom");
	(*config).sig_V = _find_value_string(config_file, "sig_V");
	(*config).main_dir = "/home/earendil/Documents/ELSA/ABM_Tactic/ABM_FINAL";//TODO: change this. //_find_value_string(config_file, "main_dir");
	return 1;
}

int get_boundary( char *bound_file, CONF_t *config ){
	FILE *rstream=fopen(bound_file, "r");
	
	if(rstream==NULL) BuG("BUG in Bound File\n");
	char c[R_BUFF];
	int Nbound,i,j;
	for(Nbound=0;fgets(c, R_BUFF, rstream);Nbound++);
	
	(*config).Nbound=Nbound;
	fclose(rstream);
	
	rstream=fopen(bound_file, "r");
	
	(*config).bound = falloc_matrix(Nbound, 2);
	for(i=0;fgets(c, R_BUFF, rstream)&&i<Nbound;i++){
		(*config).bound[i][0]=atof(c);
		for(j=0;c[j]!='\t'&&c[j]!=' ';j++);
		(*config).bound[i][1]=atof(&c[++j]);
	}
	
	fclose(rstream);
	
	if((*config).bound[0][0]!=(*config).bound[(*config).Nbound-1][0] || (*config).bound[0][1]!=(*config).bound[(*config).Nbound-1][1] )
		BuG("Please last point of the Boundary as to be the same of the first\n");
	
	return 1;
}

int get_temp_shock(CONF_t *conf){

	char* rep_tail = "/config/shock_tmp.dat";
	char * rep = malloc(snprintf(NULL, 0, "%s%s", (*conf).main_dir, rep_tail) + 1);
	sprintf(rep, "%s%s", (*conf).main_dir, rep_tail);
	
	FILE *rstream=fopen(rep,"r");
	if(rstream==NULL) BuG("Impossible to read shock_tmp.dat\n");
	
	int i,j,N;
	char c[500];
	for(N=0;fgets(c,500,rstream);N++);
	fclose(rstream);
	
	(*conf).n_point_shock = N;
	(*conf).point_shock = falloc_matrix(N, 2);
	
	rstream=fopen(rep,"r");
	for(i=0;fgets(c,500,rstream)&&i<N;i++){
		(*conf).point_shock[i][0]=atof(c);
		for(j=0;c[j]!='\t'&&c[j]!=' '&&c[j]!='\0';j++);
		if(c[j]=='\0') BuG("Error in shock_tmp.dat");
		(*conf).point_shock[i][1]=atof(&c[++j]);
	}
	fclose(rstream);
	free(rep);
	return 1;
	
}

int add_nsim_output(char *file_out,char *file_in, int n){
	int i;
	for(i=0;file_in[i]!='\0';i++);
	for(;file_in[i]!='.';i--);
	int j;
	char str[50];
	sprintf(str, "%d", n);
	
	for(j=0;j<i;j++) file_out[j]=file_in[j];
	file_out[j]='\0';
	
	strcat(file_out,"_");
	strcat(file_out,str);
	strcat(file_out,".dat");
	
	return 1;
	
}
