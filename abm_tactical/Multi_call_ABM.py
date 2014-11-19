#~ Questo e' un generico programma per il multi-process. E' necessario che la variabile cfg sia un vettore(tupla)
#~ in cui ogni riga rappresenta un thread. Poi andra' modificato il worker e cio che ne e' realtivo

import multiprocessing as ml
import time
import pylab as pl
import os

ncpu=4

in_f = lambda x: 'inputABM_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'
out_f = lambda x: 'outputABM_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'

def worker(T):
	
	input_f='DECONFLICT/'+in_f(T)
	output_f='DATA/'+out_f(T)
	

	os.system('./ElsaABM '+input_f+' '+output_f+' >> log_ABM.out 2>&1 ')
	
	return 

#----------------------------------------------------------#


def start_Multiproc(cfg):
	jobs=[]

	for i in range(len(cfg)):
		jobs.append(ml.Process(target=worker,args=[cfg[i]]))
		jobs[-1].start()
		time.sleep(1)
	
	#~ Aspetta che finiscono
	for q in jobs:
		q.join()
	
	return

#----------------------------------------------------------#

def get_configuration():
	
	#~ Original value of Eff 
	Eff=pl.linspace(0.9728914418,0.99999,20).tolist()
	Nflight=range(1500,2300,100)
	#Nflight=[1000]
	cfg=[(a,n,i) for n in Nflight for a in Eff for i in range(100) ]
	return cfg

#----------------------------------------------------------#

if __name__=='__main__':
	
	os.system('gcc *.c -o ElsaABM -lm')
		
	cfg=get_configuration()
	
	nsim=ncpu
	
	for n in range(0,len(cfg),ncpu):
		print "lanciate da",n,'->',n+nsim, ' su', len(cfg) 
		start_Multiproc(cfg[n:n+nsim])
		
		#~ Doverebbe evitare gli spariggi -- Boh?!
		if (n+ncpu)>len(cfg): nsim=len(cfg)-n
		else: nsim=ncpu

