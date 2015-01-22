#!/bin/bash
swig -python ABMtactic.i
gcc -fpic -c -I/usr/include/python2.7 -I/usr/lib/python2.7/config/ *.c 
#gcc -shared main.o ABMtactic_wrap.o -lm -o _ABMtactic.so
gcc -shared *.o -lm -o _ABMtactic.so

