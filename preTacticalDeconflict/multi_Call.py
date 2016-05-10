#~ Questo e' un generico programma per il multi-process. E' necessario che la variabile cfg sia un vettore(tupla)
#~ in cui ogni riga rappresenta un thread. Poi andra' modificato il worker e cio che ne e' realtivo

import multiprocessing as ml
import time
import pylab as pl
import create_route as cr
import os

ncpu=12

in_f = lambda x: 'inputABM_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'
in_neig = lambda x: 'neighboors_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'


#----------------------------------------------------------#

def cheak_out(file_r):
	fr=open(file_r,'r')
	x=fr.read()
	fr.close()
	x=x.split('\n')
	
	return not x[0]=='BUG'

#----------------------------------------------------------#


def worker(T):
	i=0
	while True:
		try:
			os.remove('DECONFLICT/'+in_f(T))
		except OSError:
			'do not do nothing'
		print T,i
		i+=1
		cr.new_route(T[0],T[1],T[2])
		os.system('Test_route/Test OUTPUT/'+in_f(T)+' OUTPUT/'+in_neig(T)+' DECONFLICT/'+in_f(T))
		if cheak_out('DECONFLICT/'+in_f(T)): break
	
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
	Nflight=range(1500,2600,100)
	
	cfg=[(a,n,i) for n in Nflight for a in Eff for i in range(100) ]
	return cfg

#----------------------------------------------------------#

if __name__=='__main__':
		
	os.system('gcc Test_route/*.c -o Test_route/Test -lm')
	
	cfg=get_configuration()
	
	nsim=ncpu
	
	for n in range(0,len(cfg),ncpu):
		print "lanciate da",n,'->',n+nsim, ' su', len(cfg) 
		start_Multiproc(cfg[n:n+nsim])
		
		#~ Doverebbe evitare gli spariggi -- Boh?!
		if (n+ncpu)>len(cfg): nsim=len(cfg)-n
		else: nsim=ncpu

