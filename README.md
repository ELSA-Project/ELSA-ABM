ELSA-ABM

This an Agent-Based for air traffic management purposes developped by the ELSA project. The ELSA project, which stands for "Empirically grounded agent based models for the future ATM scenario", aims at simulating the SESAR scenario.

The code is in two parts: 
 - The strategic layer, which deals with the optimization of planned trajectories and the overloading of sectors, with air companies and a network manager. 

 - The tactical layer, which deals with the resolution of potential conflicts by a air controller in "real time". 


=========== Strategic Layer ============

The code is written in python and is usable as module by doing "import abm_strategic". It provides several high level functions to build airspaces and generating traffic.

=========== Tactical Layer ============

The code is composed for now of the engine, coded in C, and a wrapper in python. The C code can be compiled this way:

gcc *.c -o ElsaABM.so -lm -shared -fPIC -I/usr/include/python2.7/ -lpython2.7

And one can use the abm_interface python code to control it.

To be expanded...