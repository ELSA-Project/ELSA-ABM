/*
 *  mUtility.c
 *  ElsaABM
 *
 *  Created by Christian Bongiorno on 06/03/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "mUtility.h"


#include<stdio.h>
#include<stdlib.h>
//#include<malloc.h>
#include<math.h>

long double **falloc_matrix(int n, int m){
	int i;
	long double **x;
	
	x=(long double**) malloc(n*sizeof(long double*));
	if(x==NULL){ printf("\nNo Memory");exit(0);}
	for(i=0;i<n;i++){
		*(x+i)=(long double*) malloc(m*sizeof(long double));
		if(*(x+i)==NULL) { printf("\nNo Memory");exit(0);}
	}
	return x;
}

int **ialloc_matrix(int n, int m){
	int i;
	int **x;
	
	x=(int **) malloc(n*sizeof(int*));
	if(x==NULL){ printf("\nNo Memory");exit(0);}
	for(i=0;i<n;i++){
		*(x+i)=(int*) malloc(m*sizeof(int));
		if(*(x+i)==NULL) { printf("\nNo Memory");exit(0);}
	}
	return x;
}

long double *falloc_vec(int i){
	long double *x;
	x=(long double*) malloc(i*sizeof(long double));
	if(x==NULL){ printf("\nNo Memory");exit(0);}
	
	return x;
}
int *ialloc_vec(int i){
	int *x;
	x=(int*) malloc(i*sizeof(int));
	if(x==NULL){ printf("\nNo Memory");exit(0);}
	
	return x;
}
void BuG(char *s){
	printf("%s",s);
	exit(0);
}
long double rad(long double degi){
	return PI*(degi/180.);
}
long double deg(long double radi){
	return (180.*radi)/PI;
}
/*From Cartesian to spherical*/
int _to_sphere(long double *x,long double *y){
	//y[0]=deg(acosl(x[2]/RH));
	//y[1]=deg(atan2l(x[1], x[0]));
	
//	y[1]=deg(atan2l(x[1], x[0]));
//	long double R=sqrtl(x[1]*x[1]+x[0]*x[0]);
//	y[0]=deg(atan2l(R, x[2]));

	y[1]=deg(atan2l(x[0], x[1]));
	y[0]=deg(asin(x[2]/RH));
	
	return 1;
}
int _to_cart(long double *a,long double *A){
//	A[0]=RH*sinl(rad(a[0]))*cosl(rad(a[1]));
//	A[1]=RH*sinl(rad(a[0]))*sinl(rad(a[1]));
//	A[2]=RH*cosl(rad(a[0]));
	A[0]=RH*cosl(rad(a[0]))*sinl(rad(a[1]));
	A[1]=RH*cosl(rad(a[0]))*cosl(rad(a[1]));
	A[2]=RH*sinl(rad(a[0]));
	
	return 1;
}

long double mean(long double *x,int N){
	int i;
	long double sum;
	for(i=0,sum=0;i<N;i++) sum+=x[i];
	return sum/N;
}

long double angle_betw_0pi(long double *a, long double *b){
	
	long double cross_x = a[1]*b[2] - a[2]*b[1];
	long double cross_y = a[2]*b[0] - a[0]*b[2];
	long double cross_z = a[0]*b[1] - a[1]*b[0];
	
	long double fcross = sqrtl(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z);
	long double fdot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
	
	return atan2l(fcross, fdot);
}

//~ 
//~ long double haversine_distance(long double *p1, long double *p2){
	//~ long double phi1=rad(p1[0]),phi2=rad(p2[0]),lam1=rad(p1[1]),lam2=rad(p2[1]);
	//~ 
	//long double x=acosl(sinl(phi1)*sinl(phi2)+cosl(phi1)*cosl(phi2)*cosl(lam2-lam1))*RH; 
	//~ //Vincent forumla
	//~ long double d_lam=lam1-lam2;
	//~ long double x= RH*atan2l(sqrtl(powl(cosl(phi2)*sinl(d_lam),2)+powl( cosl(phi1)*sinl(phi2)-sinl(phi1)*cosl(phi2)*cosl(d_lam),2)), sinl(phi1)*sinl(phi2)+cosl(phi1)*cosl(phi2)*cosl(d_lam) );
	//~ return x;
//~ 
//~ //	long double P1[3],P2[3];
//~ //	_to_cart(p1, P1);
//~ //	_to_cart(p2, P2);
//~ //	
//~ //	return angle_betw_0pi(P1, P2)*RH;
//~ }

long double haversine_distance(long double *p1, long double *p2){
	long double th1=p1[0], ph1=p1[1], th2=p2[0],  ph2=p2[1];

	long double dx, dy, dz;
	ph1 -= ph2;
	ph1 *= TO_RAD, th1 *= TO_RAD, th2 *= TO_RAD;
 
	dz = sin(th1) - sin(th2);
	dx = cos(ph1) * cos(th1) - cos(th2);
	dy = sin(ph1) * cos(th1);
	return asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * RH;
}

long double _norm(long double *a){
	return sqrtl(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}

int _cross_prod(long double *B,long double *C,long double *A){
	
	A[0]=B[1]*C[2]-B[2]*C[1];
	A[1]=B[2]*C[0]-B[0]*C[2];
	A[2]=B[0]*C[1]-B[1]*C[0];
	
	long double N=_norm(A);
	A[0]=A[0]/N;
	A[1]=A[1]/N;
	A[2]=A[2]/N;
	
	return 1;
}

long double _dot(long double *a,long double *b){
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}


void q_sort(long double **arr, int beg, int end){
	long double *temp;
	if (end > beg + 1){
		long double piv = arr[beg][0];
		int l = beg + 1, r = end;
		while (l < r){
			if (arr[l][0] <= piv)
				l++;
			else{
				temp=arr[l];
				arr[l]=arr[--r];
				arr[r]=temp;
			}
		}
		temp=arr[beg];
		arr[beg]=arr[--l];
		arr[l]=temp;
		
		q_sort(arr, beg, l);
		q_sort(arr, r, end);
	}
}
long double euclid_dist2d(long double *a,long double *b){
	return sqrtl(powl(a[0]-b[0],2)+powl(a[1]-b[1], 2));
}


long double euclid_dist(long double *a,long double *b){
	return sqrtl(powl(a[0]-b[0],2)+powl(a[1]-b[1], 2)+powl(a[2]-b[2], 2));
}

long double _min(long double a,long double b){
	if(a>b) return b;
	else return a;
}
long double _max(long double a,long double b){
	if(a<b) return b;
	else return a;
}
int segments_intersect(long double *a,long double *b, long double *c, long double *d){
	
	long double p[4];
	find_intersection(a, b, c, d, p);
		
	//long double ya=a[1],yb=b[1],yc=c[1],yd=d[1];
	if(isbetween(a, b, p) && isbetween(c, d, p)) return 1;
	else return 0;

//	if(xa==xb)
//		if(_min(xc,xd)<=xa && _max(xc,xd)>=xa && _min(yc,yd)<=_max(ya,yb) && _max(yc,yd)>= _min(ya,yb) )
//			return 1;
//	
//	if(xc==xd)
//		if(_min(xa,xb)<=xc && _max(xa,xb)>=xc && _min(ya,yb)<=_max(yc,yd) && _max(ya,yb)>= _min(yc,yd) )
//			return 1;
	
//	if(p[0]<=_max(xa,xb) && p[0]>=_min(xa,xb) && p[1]<=_max(ya,yb) && p[1]>=_min(ya,yb) )
//		if(p[0]<=_max(xc,xd) && p[0]>=_min(xc,xd) && p[1]<=_max(yc,yd) && p[1]>=_min(yc,yd) ) return 1;
//	if( p[1]<=_max(ya,yb) && p[1]>=_min(ya,yb) )
//			if( p[1]<=_max(yc,yd) && p[1]>=_min(yc,yd) ) return 1;
	
	//if(p[2]<=_max(xa,xb) && p[2]>=_min(xa,xb) && p[2]<=_max(xc,xd) && p[2]>=_min(xc,xd) ) return 1;
	
	return 0;
}
long double _den(long double a,long double b,long double c,long double d,long double e,long double f){
	//return sqrtl( (c*c*(d*d+e*e)-2*a*c*d*f-2*b*e*(a*d+c*f)+b*b*(d*d+f*f)+a*a*(e*e+f*f))/powl(b*d-a*e,2) );
	return sqrtl(1.+(c*c*d*d + c*c*e*e - 2*a*c*d*f - 2*b*c*e*f + a*a*f*f + b*b*f*f)/powl(b*d-a*e, 2));
}


long double _maxim(long double **poly,int N){
	int i;
	long double M;
	for(i=1,M=poly[0][0];i<N;i++) if(poly[i][0]>M) M=poly[i][0];
	
	return M;
}

/* Find the intersection between two rect passing to four point*/
int find_intersection(long double *p0,long double *p1,long double *p2,long double *p3,long double *p){
	
	/* SUL CERCHIO */
	long double P0[3],P1[3],P2[3],P3[3];
	_to_cart(p0,P0);
	_to_cart(p1,P1);
	_to_cart(p2,P2);
	_to_cart(p3,P3);
	
	long double a = (P0[1]*P1[2] - P1[1]*P0[2] );
	long double b = -(P0[0]*P1[2] - P1[0]*P0[2] );
	long double c = (P0[0]*P1[1] - P1[0]*P0[1] );
	long double d = (P2[1]*P3[2] - P3[1]*P2[2] );
	long double e = -(P2[0]*P3[2] - P3[0]*P2[2] );
	long double f =  (P2[0]*P3[1] - P3[0]*P2[1] );
		
	//Parametric version of plane (with conditions of ortogonality)*/
	long double X1[3],X2[3];
	
	long double t=RH/_den(a,b,c,d,e,f);
	
	if(_den(a,b,c,d,e,f)<SGL) BuG("DEN ZERO\n");
	if(fabsl(b*d-a*e)<SGL) BuG("bd ae zero\n");
	
	X1[0]= -((b*f-c*e)/(b*d-a*e))*t;
	X1[1]= -((c*d-a*f)/(b*d-a*e))*t;
	X1[2]=t;

	X2[0]=-X1[0];
	X2[1]=-X1[1];
	X2[2]=-X1[2];
		
	_to_sphere(X1,p);
	_to_sphere(X2,&p[2]);

	if(euclid_dist(P0,X1)>euclid_dist(P0,X2)) {
		long double comodo;
		comodo=p[2];
		p[2]=p[0];
		
		comodo=p[3];
		p[3]=p[1];
	}

	return 1;
}

int point_in_polygon(long double *p,long double **poly,int Np){
	int inter,i;
	
	long double M[]={_maxim(poly,Np),p[1]};
	
	for(i=0,inter=0;i<(Np-1);i++) inter+=segments_intersect(p,M,poly[i],poly[i+1]);
	
	if(inter%2==0) return 0;
	
	else return 1;
}

long double frand(long double fMin,long double fMax){
	long double f = (long double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

int	 irand(int max){
	return (int)((rand()/(RAND_MAX +1.0))*max);
}

int eucl_isbetween(long double *a,long double *b, long double *p){
	long double x = a[0]*b[1] + a[1]*p[0] + b[0]*p[1] - p[1]*a[0] - b[0]*a[1] - p[0]*b[1];
	
	if( x<0.1 ) return 1;
	else return 0;
	
}
//ritorna 1 se x compreso nella retta passante per a,b
int isbetween(long double *a,long double *b, long double *x){

	if(fabsl(x[0]-a[0])<SGL && fabsl(x[1]-a[1])<SGL) return 1;
	if(fabsl(x[0]-b[0])<SGL && fabsl(x[1]-b[1])<SGL) return 1;
	
	long double P0[3],P1[3],P2[3];
	_to_cart(a, P0);
	_to_cart(b, P1);
	_to_cart(x, P2);
	
	long double S[3];
	_cross_prod(P0, P1, S);
	long double gc = S[0]*P2[0] +S[1]*P2[1] + S[2]*P2[2];

	if(fabsl(gc)>SGL) return 0;
	
	long double ax=angle_betw_0pi(P0, P2);
	long double bx=angle_betw_0pi(P1, P2);
	long double ab=angle_betw_0pi(P0, P1);
	
	if(ax<=ab && bx <= ab) return	 1;
	else return 0;
}
void freeTouch(long double **f){
	if(*f != NULL){
		free(*f);
		*f = NULL;
	}
}
void freeTouch2(long double ***f){
	if(*f != NULL){
		free(*f);
		*f = NULL;
	}
}
void ifreeTouch(int **f){
	if(*f != NULL){
		free(*f);
		*f = NULL;
	}
}
void ifreeTouch2(int ***f){
	if(*f != NULL){
		free(*f);
		*f = NULL;
	}
}


void ffree_2D(long double **x,int N){
	int i;
	for(i=0;i<N;i++) {
		//printf("%d\t %Lf\n",i,x[i][0]);
		freeTouch((x+i));
	}
	//printf("Qui\n");
	freeTouch2(&x);
	return;
}

void ifree_2D(int **x,int N){
	int i;
	for(i=0;i<N;i++) {
		//printf("%d\t %Lf\n",i,x[i][0]);
		ifreeTouch((x+i));
	}
	//printf("Qui\n");
	ifreeTouch2(&x);
	return;
}

/*
void ifree_2D(int **x,int N){
	int i;
	for(i=0;i<N;i++) 
		free((x[i]));
	free(x);
}*/
void ffree_3D(long double ***x,int N,int M){
	int i,j;
	for(i=0;i<N;i++) for(j=0;j<M;j++) free(*(*(x+i)+j));
	for(i=0;i<N;i++) free(*(x+i));
	free(x);
}


int find_st_indx(long double **nvp,int n_nvp,long double *st_point,int st_indx,long double *old_st_point){
	int i,N=n_nvp-1;
	
	#ifdef EUCLIDEAN
	if(st_indx!=0) if(eucl_isbetween(old_st_point,nvp[st_indx],st_point)) return st_indx;
	for(i=st_indx;i<N;i++) {
		if(eucl_isbetween(nvp[i],nvp[i+1],st_point)) return i+1;
	}
	#else
	if(st_indx!=0) if(isbetween(old_st_point,nvp[st_indx],st_point)) return st_indx;
	for(i=st_indx;i<N;i++) {
		if(isbetween(nvp[i],nvp[i+1],st_point)) return i+1;
	}
	#endif
	return -1;
}

int find_p_indx(long double **nvp,int n_nvp,long double *p){
	int i;
	#ifdef EUCLIDEAN
	for(i=0;i<(n_nvp-1);i++) if(eucl_isbetween(nvp[i],nvp[i+1],p)) return i+1;
	#else
	for(i=0;i<(n_nvp-1);i++) if(isbetween(nvp[i],nvp[i+1],p)) return i+1;	
	#endif
	//BuG("Impossible to find point\n");
	
	return 0;	
}
void mischia(int *carte,int N){
	int i,comodo,x;
	
	for(i=0;i<N;i++) carte[i]=i;
	for(i=(N-1);i>0;i--){
		x=irand(i+1);
		comodo=carte[i];
		carte[i]=carte[x];
		carte[x]=comodo;
	}
	return;
}

int eucl_coord(long double vel,long double *pa,long double *pb,long double *p,long double t){
	
	long double D = euclid_dist2d(pb,pa);
	long double d = vel*t;
	
	p[0] = pa[0] + d*(pb[0]-pa[0])/D;
	p[1] = pa[1] + d*(pb[1]-pa[1])/D;
	
	return 1;

}

int coord(long double vel,long double *pa,long double *pb,long double *p,long double t){
	
	long double A[3],B[3],P[3];
	_to_cart(pa, A);
	_to_cart(pb, B);
	
	long double AB=angle_betw_0pi(A, B);
	long double AP=(vel*t)/RH;

	long double cos_ap=cosl(AP);
	long double sin_rap= sinl(AP)/(sinl(AB)*RH*RH);
	
	P[0] = A[0]*cos_ap + ( +A[1]*A[1]*B[0] + A[2]*A[2]*B[0] - A[0]*A[1]*B[1] - A[0]*A[2]*B[2] )*sin_rap;
	
	P[1] = A[1]*cos_ap + (- A[0]*A[1]*B[0] + A[0]*A[0]*B[1] + A[2]*A[2]*B[1] - A[1]*A[2]*B[2] )*sin_rap;

	P[2] = A[2]*cos_ap + (-A[0]*A[2]*B[0] - A[1]*A[2]*B[1] + A[0]*A[0]*B[2] + A[1]*A[1]*B[2] )*sin_rap;
		
	_to_sphere(P, p);

	return 1;
																								
	
}

//~ int coord3(long double vel, long double beta, long double *pos, long double t,long double *p){
	//~ long double a=(vel*t)/RH;
	//~ long double b=asinl(sinl(a)*sinl(beta));
	//~ 
	//~ long double c=2*atan2l(tanl(0.5*(a-b))*sinl(0.5*(PI/2+beta)), sinl(0.5*(PI/2-beta)));
	//~ if(c>PI/2) c=2*PI -c;
	//~ p[0]=pos[0]+deg(c);
	//~ p[1]=pos[1]+deg(b);
		//~ 
	//~ return 0;
//~ }
long double eucl_angle_direction(long double *a,long double *b, long double *c){
	long double Ap[2];
	long double Cp[2];
	
	Ap[0] = a[0]-b[0];
	Ap[1] = a[1]-b[1];
	
	Cp[0] = c[0]-b[0];
	Cp[1] = c[1]-b[1];
	
	long double DAp = atan2l(Ap[1],Ap[0]);
	long double DCp = atan2l(Cp[1],Cp[0]);
	
	return DAp - DCp ;
	//return  2*PI - DAp - DCp ;

}

long double angle_direction(long double *a,long double *b, long double *c){
	//	long double alpha=rad(q[1]-p[1]), beta=rad(q[0]-p[0]);
	//
	//	return atan2l(sinl(alpha), sinl(beta)*cosl(alpha));
	long double A[3],B[3],C[3];
	
	_to_cart(a,A);
	_to_cart(b, B);
	_to_cart(c, C);
	
	long double N1[3],N2[3];
	
	_cross_prod(A, B, N1);
	_cross_prod(B, C, N2);
	
	return angle_betw_0pi(N1, N2);
	
}


/*Find the point c of minimum distance in the circle ab*/
int proj_point_to_circle(long double *p0,long double *p1,long double *px,long double *ph,long double *dist){
	
	long double A[3],B[3],P[3];
	_to_cart(p0, A);
	_to_cart(p1, B);
	_to_cart(px, P);
	
	/*Calculate the ortogonal vector to AB*/
	long double N[3];
	_cross_prod(A,B,N);
	
	/*Calculate the angle between Ortognoal Vetctor and point P vector (distance)*/
	//long double theta=  acosl(_dot(N,P)/ (_norm(N)*_norm(P)) )-PI/2. ;
	long double theta=  angle_betw_0pi(N, P) -PI/2. ;
	
	*dist=RH*fabsl(theta);
	
	/*Calculate director vector of plane NPT*/
	long double T[3];
	_cross_prod(N, P, T);
	
	long double a=N[0];
	long double b=N[1];
	long double c=N[2];
	long double d=T[0];
	long double e=T[1];
	long double f=T[2];
	
	/*Calculate value of intersection under the condition i.e. t parameters*/
	long double t=RH/_den(a,b,c,d,e,f);
	
	if(_den(a,b,c,d,e,f)==0) BuG("DEN ZERO\n");
	if((b*d-a*e)==0) BuG("bd ae zero\n");

	//Parametric version of plane (with conditions of ortogonality)*/
	long double X1[3],X2[3];
	X1[0]= -((b*f-c*e)/(b*d-a*e))*t;
	X1[1]= -((c*d-a*f)/(b*d-a*e))*t;
	X1[2]=t;
	
	X2[0]=-X1[0];
	X2[1]=-X1[1];
	X2[2]=-X1[2];
	
	if(euclid_dist(X1, A)< euclid_dist(X2, A)) _to_sphere(X1,ph);
	else _to_sphere(X2,ph);
	
	return 1;
}

long double distance_point_segment(long double *a,long double *b,long double *p){
	long double d,c[2];
	
	proj_point_to_circle(a, b, p, c, &d);
	
	if(c[0]>=_min(a[0], b[0]) && c[0]<=_max(a[0], b[0]) && c[1]>=_min(a[1], b[1]) && c[1]<=_max(a[1], b[1])) return d;
	
	long double d2=haversine_distance(a, p);
	d=haversine_distance(b, p);
	
	if(d2<d) return d2;
	else return d;
}

int time_to_int(long double time,int *T){

	T[0] = (int)(time /3600.);
	T[1] = (int)((time - T[0]*3600.)/60.);
	T[2] = (int) (time -T[0]*3600. -T[1]*60.);
	T[3] = (int) ((time -T[0]*3600. -T[1]*60.-(long double)T[2])*1000000.);
						
	return 1;
}
