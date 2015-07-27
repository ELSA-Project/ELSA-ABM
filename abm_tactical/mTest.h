/*
 *@author: cbongiorno.
 */

#ifndef __MTEST_H
#define __MTEST_H


/* To use strptime and other Time features*/
#define __USE_XOPEN 
#define  _GNU_SOURCE


void plot(Aircraft_t ,CONF_t ,char*);

void print_tmp_point(CONF_t);

void plot_ID(Aircraft_t* ,int ,CONF_t ,int);

void plot_pos(Aircraft_t,CONF_t,char*);

void plot_movie(Aircraft_t **f,int, CONF_t,char *);

void plot_bound(CONF_t);

void print_nvp(Aircraft_t );

void cheak_inside_pos(Aircraft_t *,int ,CONF_t );

void plot_where(Aircraft_t ,CONF_t ,char *);

void plot_pos_gall(Aircraft_t ,CONF_t ,char *);

void gall_peter(long double *,long double *);

void print_time(long double );

void plot_point(long double *,char *);

int cheak_nan_pos(Aircraft_t *,CONF_t );

//void print_workload(TOOL_f , CONF_t , char *);

#endif
