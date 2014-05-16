# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:48:19 2013

@author: luca
"""

#import matplotlib.pyplot as plt

#import sys
#sys.path.insert(1,'../Modules')
import matplotlib.pyplot as plt
import numpy as np
from iter_simO import build_path
import pickle as pk
import ABMvars

#from iter_simO import make_file_name
#plt.figure(1)
#plt.title(str(ABMvars.N)+' nodes - Graph type '+str(ABMvars.type_of_net)+' - Nfp = '+str(ABMvars.Nfp)+' - airport distance = '+str(ABMvars.airport_dis)+' (edges)')
#plt.xlabel('order of rejection')
#plt.ylabel('average time of the I rejection')

version='2.9.0'

#def build_path(paras, loop_on='par', in_title=['N', 'Nfp', 'sigma', 'tau', 'par']):
#    """
#    Used to build a path from a set of paras. 
#    New in 2.2.
#    """
#    name=build_path_single(paras,in_title=[p for p in in_title if p!=loop_on])
#    name+='_loop_on_' + loop_on
#    name+= '_iter'+ str(paras['n_iter'])
#        
#    return name

class Plot:
    """
    Class Plot
    ==========
    Used to post-process the results and make the plots.
    """
    def __init__(self,paras):
        self.paras=paras
        self.rep=build_path(paras)
        self.results_processed={}
        self.colors=('Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed')
    
    def import_results(self,rep=''):
        """
        Used to import results from the disk.
        """
        print 'Importing results...',
        if rep=='':
            rep=self.rep
        try:
            f=open(rep + '/results.pic','r')
            self.results=pk.load(f)
            f.close()
        except IOError:
            print 'File not found!'
            print 'I was looking for', rep + '/results.pic'
            raise
        except:
            raise
            
#        for set_par in self.results.keys():
#            self.results_processed[set_par]={}
            
        print ' done.'
#    def post_process_results(self, save=True):
#        """
#        Used to post-process results. Every processes between the simulation and 
#        the plots should be here.
#        Changed in 2.4: add satisfaction, regulated flight & regulated flight plans. On level of iteration added (on par).
#        """
#        print 'Post-processing results...',
#        for set_par in self.results.values():
#            for it in set_par:
#                for f in it:   
#                    # Make flags
#                    f.make_flags()
#                    
#                    #Satisfaction
#                    bestcost=f.FPs[0].cost
#                    acceptedFPscost=[FP.cost for FP in f.FPs if FP.accepted]
#                    if len(acceptedFPscost)!=0:
#                        f.satisfaction=bestcost/min(acceptedFPscost)
#                    else:
#                        f.satisfaction=0                    
#                        
#                    # Regulated flight plans
#                    f.regulated_FPs=len([FP for FP in f.FPs if not FP.accepted])
#                    
#                    # Regulated flights
#                    f.regulated_F = len([FP for FP in f.FPs if FP.accepted])==0
#        print ' done.'
            
                 

                
    def compute_times(self):
        """
        For now, computes the average and standard deviation of the first 
        time of rejection.
        """
        #times={p:{} for p in results}
        self.times={}
        for p,sim in self.results.items():
            self.times[p]={'avg':[],'std':[]}
            for i in range(self.paras['Nfp']):
                t=[]
                for flights in sim:
                    flags=np.array([f.flag_first for f in flights])
                    try:
                        t.append(np.where(flags>i)[0][0])
                    except IndexError:
                        pass
                    except:
                        raise
                self.times[p]['avg'].append(np.mean(t))
                self.times[p]['std'].append(np.std(t))
    
    def rejmeans(self):
        """
        Make plots of the first time of rejection.
        """
        if not hasattr(self, 'times'):
            self.compute_times()
        
        plt.figure(1)
        plt.title(str(self.paras['N'])+' nodes - Graph type '+str(self.paras['type_of_net'])+' - Nfp = '+str(self.paras['Nfp']))
                #+' - airport distance = '+ str(self.paras['airport_dis'])+' (edges)')
        plt.xlabel('order of rejection')
        plt.ylabel('average time of the I rejection')
        

        #plt.xlim(0,self.paras['Nfp'])
        for i,p in enumerate(self.times.keys()):
            res=self.times[p]
            
            if self.paras['para_iter']=='par':            
                plt.errorbar(range(self.paras['Nfp']),res['avg'],res['std'],color=self.colors[i%len(self.colors)],fmt='--o',\
                            label=r'$\alpha=$'+str(p[0][0])+' '+r'$\beta=$'+str(p[0][2]))
            elif self.paras['para_iter']=='tau':
                plt.errorbar(range(self.paras['Nfp']),res['avg'],res['std'],color=self.colors[i%len(self.colors)],fmt='--o',\
                            label=r'$\tau=$'+str(p))
        plt.legend(loc=2)
        plt.savefig(self.rep + '/rejmeans.png')
        plt.show()
        
    def satisfaction(self):
        """
        Make plots an satisfaction and Co.
        """
        print 'Doing satisfaction plots...',
        plt.figure(1)
        plt.title('')
        plt.xlabel('alpha/beta')#self.paras['para_iter'])
        plt.ylabel('Global Satisfaction')

        if self.paras['para_iter']=='par':
            results_to_plot={}
            for k,v in self.results.items():
                print k, float(k[0][0])/float(k[0][2])
                results_to_plot[float(k[0][0])/float(k[0][2])]=v
        else:
            results_to_plot=self.results
        
        keys=sorted(results_to_plot.keys())

        plt.semilogx(keys,[results_to_plot[k]['global_satisfaction'] for k in keys],'ro-')
        plt.savefig(self.rep + '/global_satisfaction.png')
        
        plt.show()
        print ' done.'
        
    def main(self):
        self.import_results()
        self.post_process_results()
        #plot.rejmeans()
        self.do_global_metrics()       
        self.satisfaction()



if __name__=='__main__':

    plot=Plot(ABMvars.paras)
    plot.main()
    
   
