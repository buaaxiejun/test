import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loaddata
def loaddata(filename):
    fr = open(filename)
    x = []
    y = []
    for line in fr.readlines():
        line = line.strip().split()
        x.append([float(line[0]),float(line[1])])
        y.append(float(line[-1]))
    return np.mat(x), np.mat(y)
x, y = loaddata("data1.txt")
print(y.shape)
print(x.shape)
print(y)
print(x)
y = y.tolist()
x = np.array(x)
print(y)
color = []
for i in range(len(y[0])):
    if y[0][i] == 1.0:
        color.append("red")
    else:
        color.append("green")
print(color)
plt.scatter(x[:,0],x[:,1],c = color)
plt.show()


#数据归一化
def scalling(data):
    max = np.max(data, 0)
    min = np.min(data, 0)
    return (data - min) / (max - min) ,max, min

xmat_s = scalling(x)
# print(xmat_s)

#sigmoid
def sigmoid(data):
    return 1 / (1 + np.exp(-data))

#w b ots
def wb_calc(X,ymat,alpha = 0.01,maxstep=10000,n_hidden_dim = 3,reg_lambda=0):
    W1 = np.mat(np.random.randn(2,n_hidden_dim))
    b1 = np.mat(np.random.randn(1,n_hidden_dim))
    W2 = np.mat(np.random.randn(n_hidden_dim,1))
    b2 = np.mat(np.random.randn(1, 1))
    w1_save = []
    w2_save =[]
    b2_save = []
    b1_save = []
    ss = []
    # FP
    for step in range(maxstep):
        z1 = X*W1 + b1  #(20,2)(2,3) + (1,3) = (20,3)
        a1 = sigmoid(z1)  #(20,3)
        z2 = a1*W2 + b2   #(20,3)(3,1) + (1,1) = (20,1)
        a2 = sigmoid(z2)  #(20,1)
        # print(a2.shape)
        # print(ymat.shape)

        #BP
        a0 = X.copy()
        deta2 = a2 - ymat  #(20,1)
        # print(deta2.shape)
        deta1 = np.mat((deta2 * W2.T).A * (a1.A*(1-a1).A))
        #(20,1)(1,3).*(20,3) = (20,3)
        dW1 = a0.T*deta1 + reg_lambda*W1  #(2,20)(20,3) + (2,3) +(2,3)
        db1 = np.sum(deta1,0)   #?
        #db2 = np.mat(np.ones()) * deta1
        dW2 = a1.T*deta2 + reg_lambda * W2
        db2 = np.sum(deta2,0)
        #updata w b
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        if step % 200 == 0:
            w1_save.append(W1.copy())
            w2_save.append(W2.copy())
            b2_save.append(b2.copy())
            b1_save.append(b1.copy())
            ss.append(step)
    return W1,b1,W2,b2,w1_save,b1_save,w2_save,b2_save, ss
    # print("W1:" ,W1)
    # print("W2:", W2)
    # print("b1:", b1)
    # print("b2:", b2)
xmat,ymat = loaddata("data1.txt")
xmat_s,xmat_max,xmat_min= scalling(xmat)
W1,b1,W2,b2,w1_save,b1_save,w2_save,b2_save, ss= wb_calc(xmat_s,ymat.T,0.05,30000,10,0)
ymat = ymat.T

#show

plotx1 = np.arange(0,10,0.01)  #array
plotx2 = np.arange(0,10,0.01)
plotX1,plotX2 = np.meshgrid(plotx1,plotx2)
plotx_new = np.c_[plotX1.ravel(),plotX2.ravel()]
plotx_new2 = (plotx_new - xmat_min) / (xmat_max - xmat_min)

for i in range(len(w1_save)):
    plt.clf()
    plot_z1 = plotx_new2*w1_save[i] + b1_save[i]
    plot_a1 = sigmoid(plot_z1)
    plot_z2 = plot_a1*w2_save[i] + b2_save[i]
    plot_a2 = sigmoid(plot_z2)
    ploty_new = np.reshape(plot_a2,plotX1.shape)
    print(ploty_new)
    plt.contourf(plotX1, plotX2,ploty_new,1,alpha=0.5)
    plt.scatter(xmat[:,0][ymat==0].A,xmat[:,1][ymat==0].A,s=100,marker="o",label="0")
    plt.scatter(xmat[:,0][ymat==1].A,xmat[:,1][ymat==1].A,s=150,marker="^",label="0")
    plt.grid()
    plt.legend()
    plt.title(f"iter:{ss[i]}")
    plt.pause(0.01)
plt.show()

"""
learn how to use git to upload codes to github
"""

