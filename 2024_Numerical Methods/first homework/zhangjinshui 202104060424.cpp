/******************************************************************************

                            Online C Compiler.
                Code, Compile, Run and Debug C program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <stdio.h>
#include <iostream>
//车辆2101 张金水 202104060424
//定义拉格朗日函数
double Lagrang(double* x,double* y,double m,int n){//此处定义x,y是双精度的，其中，m是随机输入自变量，n是阶次
    if((sizeof(x)/sizeof(x[0]))!=(sizeof(y)/sizeof(y[0]))){
        std::cout <<"发现错误，x,y数组的长度不一"<< std::endl;//保证函数有解
        return 9999999999999;
    }
    if(sizeof(x)/sizeof(x[0])>n){//保证函数的阶次
        return 9999999999999;
    }
    double* l=new double[n+1];//建立x[k]处的Lagrange的插值函数l_k(x[i])
    for (int i=0;i<=n;i++){
        double z=1;
        for(int j=0;j<=n;j++){
            if(j==i){
                continue;
            
            }
            z=z*(m-x[j])/(x[i]-x[j]);
            
        }
        l[i]=z;
    }
    //计算函数值L_n(x[i])等于y[k]*l_k(x[i])的叠加
    double Ln=0;
    for(int i=0;i<=n;i++){
        Ln=Ln+l[i]*y[i];
    }
    return Ln;
}//代入数值运算。此处以2阶为例
int main(){
    double x[]={1,2,3,4};
    double y[]={4.526427,4.578543,4.263369,4.123321};
    int n=2;
    double m=2.36;
    std::cout << "预测值为：" << Lagrang(x,y,m,n) << std::endl;
    return 0;
}

