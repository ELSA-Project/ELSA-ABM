ABM-Tactical-Layer
==================

This an Agent-Based for air traffic management purposes developped by the ELSA project. 
The ELSA project, which stands for "Empirically grounded agent based models for the future ATM scenario", aims 
at simulating the future SESAR scenario.

The code is composed for now of the engine, coded in C, and a wrapper in python. The C code can be 
compiled this way:

gcc *.c -o ElsaABM.so -lm -shared -fPIC -I/usr/include/python2.7/ -lpython2.7 

And one can use the abm_interface python code to control it.

