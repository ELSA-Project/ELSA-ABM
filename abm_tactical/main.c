#include "mQuery.h"
#include "mSector.h"
#include "mUtility.h"
#include "mABM.h"
#include "mTest.h"
//#include <Python.h>

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// Entry for wrapper.
int simulation(char **args){

	srand(time(NULL));
	
	CONF_t config;
	Aircraft_t *Flight,*flight;
	int Nflight;
	SHOCK_t shock;
	
	char *input_ABM=args[1];
	char *output_ABM=args[2];
	char *config_file=args[3];
	char output_ABM_nsim[400]; //TODO: change this!
	
	init_Sector(&Flight,&Nflight,&config,&shock,input_ABM, config_file);
	
	int i;
	for(i=0;i<config.nsim;i++) {
		
		copy_flight(Flight,Nflight,&flight);
		
		
		if( ABM(&flight,Nflight,config,shock) == 0){
			del_flight(&flight, Nflight, Flight);
			i--;
			continue;
		}
		printf("Sim %d\n",i+1);
		add_nsim_output(output_ABM_nsim,output_ABM,i);
		save_m3(flight,Nflight,Flight,output_ABM_nsim);
		
		del_flight(&flight, Nflight, Flight);
	}
	
	del_flight_pos(&Flight,Nflight,config);
	del_flight(&Flight, Nflight, Flight);
	del_conf(&config);
	del_shock(&shock);
	
	return 0;
}

// Manual entry
int main(int argc,char *argv[]){
	simulation(argv);
}


