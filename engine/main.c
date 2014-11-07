#include "mQuery.h"
#include "mSector.h"
#include "mUtility.h"
#include "mABM.h"
#include "mTest.h"
#include <Python.h>

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(int argc,char *argv[]){
	srand(time(NULL));
	
	CONF_t config;
	Aircraft_t *Flight,*flight;
	int Nflight;
	SHOCK_t shock;
	
	char *input_ABM=argv[1];
	char *output_ABM=argv[2];
	char output_ABM_nsim[100];
	
	init_Sector(&Flight,&Nflight,&config,&shock,input_ABM);
	
	//init_output(Flight,Nflight);
	
	int i;
	for(i=0;i<config.nsim;i++) {
		
		copy_flight(Flight,Nflight,&flight);
		printf("Sim %d\n",i+1);
		ABM(&flight,Nflight,config,shock);
		
		add_nsim_output(output_ABM_nsim,output_ABM,i);
		save_m3(flight,Nflight,Flight,output_ABM_nsim);
		
		del_flight(&flight, Nflight,Flight);
		
	}
	
	
	
	return 0;
	
}

static PyObject* py_iter_sim(PyObject* self, PyObject* args)
{
	char x, y;
	char **argv;
	PyArg_ParseTuple(args, "ss", &x, &y);
	argv[0] = x;
	argv[1] = y;
	main(2, argv);
	return Py_BuildValue("d", 0.);
}


/*
 * Bind Python function name to C function
 */
static PyMethodDef ElsaABM_methods[] = {
	{"iter_sim", py_iter_sim, METH_VARARGS},
	{NULL}
};

/*
 * Python calls this to initialize the module
 */
void initElsaABM()
{
	(void) Py_InitModule("ElsaABM", ElsaABM_methods);
}
