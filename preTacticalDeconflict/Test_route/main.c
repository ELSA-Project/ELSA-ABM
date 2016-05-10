#include "mQuery.h"
#include "mSector.h"
#include "mUtility.h"
#include "mABM.h"
#include "mTest.h"

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(int argc,char *argv[]){
	srand(time(NULL));
	
	CONF_t config;
	Aircraft_t *Flight,*flight;
	int Nflight;
	SHOCK_t shock;
	
	char *input_file=argv[1];
	char *neigh_file=argv[2];
	char *output_file=argv[3];
	
	init_Sector(&Flight,&Nflight,&config,&shock,input_file,neigh_file);
	
	//init_output(Flight,Nflight);
	
	//copy_flight(Flight,Nflight,&flight);
	ABM(&Flight,Nflight,config,shock,output_file);
		
	//del_flight(&flight, Nflight,Flight);
		
	
	
	
	
	return 0;
	
}
