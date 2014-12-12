#include "mQuery.h"
#include "mSector.h"
#include "mUtility.h"
#include "mABM.h"
#include "mTest.h"
//#include <Python.h>

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int simulation(char **args){

	srand(time(NULL));
	
	CONF_t config;
	Aircraft_t *Flight,*flight;
	int Nflight;
	SHOCK_t shock;
	
	char *input_ABM=args[1];
	char *output_ABM=args[2];
	char output_ABM_nsim[100];
	
	init_Sector(&Flight,&Nflight,&config,&shock,input_ABM);
	
	int i;
	for(i=0;i<config.nsim;i++) {
		
		copy_flight(Flight,Nflight,&flight);
		printf("Sim %d\n",i+1);
		ABM(&flight,Nflight,config,shock);
		
		add_nsim_output(output_ABM_nsim,output_ABM,i);
		save_m3(flight,Nflight,Flight,output_ABM_nsim);
		
		del_flight(&flight, Nflight, Flight);
	}
	return 0;
}


int main(int argc,char *argv[]){
	simulation(argv);
}


