#include <stdio.h>
#include <stdlib.h>

const int N=4096;

int main(){
	double a[N],b[N],c[N];
	for(int i=0;i<N;++i) a[i]=1,b[i]=3,c[i]=1;

	for(int i=0;i<N;++i)
		c[i]=a[i]*b[i]+c[i];
	double sum=0;
	for(int i=0;i<N;++i)
		sum+=c[i];
	printf("%f\n",sum);
}