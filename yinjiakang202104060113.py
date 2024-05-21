'''

Welcome to GDB Online.尹家康202104060113
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
def coefficient(vari, k, i, n):
    if i >= n - 1:
        if i > n-1:
            return 1
        elif k != n-1:
            return (vari - x[n-1]) / (x[k] - x[n-1])
        else:
            return 1
    elif i != k:
        return (vari - x[i]) / (x[k] - x[i]) * coefficient(vari, k, i+1, n)
    else:
        return (vari - x[i+1]) / (x[k] - x[i+1]) * coefficient(vari, k, i+2, n)
a = input("请输入若干个互异的节点的x轴坐标值：")
x = [float(i) for i in a.split(" ")]
b = input("请输入若干个互异的节点的y轴坐标值：")
y = [float(j) for j in b.split(" ")]
n = len(x)
sum = 0
v = float(input("请输入需要计算的节点的x轴坐标值："))

for k in range(n):
    increase = (y[k]) * coefficient(v, k, 0, n)
    print("系数为:"+str(increase))
    sum += increase

print("x=%f, y=%f" % (v, sum))