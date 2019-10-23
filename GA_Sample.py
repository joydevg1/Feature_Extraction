# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 22:26:26 2018

@author: Joydev Ghosh
"""
import csv
import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import operator
counttcp=0
counticmp=0
countudp=0
data=open("20 Percent Training Set new.csv","r")
X=list(csv.reader(data))
#Preprocessing of target values
value=open("Target_values.csv","r")
Y=list(csv.reader(value))
Z1=[]
for i in range(0,len(Y)):
    if Y[i]:
        Z1.append(Y[i])
out=np.ravel(np.array(Z1))       

test_value=open("test_target.csv","r")
Y_test=list(csv.reader(test_value))
Z1_test=[]
for i in range(0,len(Y_test)):
    if Y_test[i]:
        Z1_test.append(Y_test[i])
out_test=np.ravel(np.array(Z1_test))
'''targettemp=[]
for i in range(0,len(out)):
    for j in range(0,len(targetnew1)):
        if out[i]==targetnew1[j][0]:
            targettemp.append(targetnew1[j][1])
for i in range(0,len(targettemp)):
    if targettemp[i]=='normal':
        targettemp[i]=1
    if targettemp[i]=='r2l':
        targettemp[i]=2
    if targettemp[i]=='u2r':
        targettemp[i]=3
    if targettemp[i]=='dos':
        targettemp[i]=4
    if targettemp[i]=='probe':
        targettemp[i]=5
    if targettemp[i]=='unknown':
        targettemp[i]=0
print(targettemp[3173])
'''
#Preprocessing of features
print(len(X[0]))
for i in range(0,len(X)):
    for j in range(0,len(X[0])):
        if X[i][j]=='tcp':
            counttcp=counttcp+1
        elif X[i][j]=='icmp':
            counticmp=counticmp+1
        elif X[i][j]=='udp':
            countudp=countudp+1
print(counticmp)
for i in range(0,len(X)):
    for j in range(0,len(X[0])):
        if X[i][j]=='tcp':
            X[i][j]=format(counttcp/(counttcp+counticmp+countudp),'.10f')
        elif X[i][j]=='icmp':
            X[i][j]=format(counticmp/(counttcp+counticmp+countudp),'.10f')
        elif X[i][j]=='udp':
            X[i][j]=format(countudp/(counttcp+counticmp+countudp),'.10f')
Z=np.array(X)
uniq=np.unique(Z[:,2])
new=[0]*len(uniq)
for i in range(0,len(Z)):
    for j in range(0,len(uniq)):
        if Z[i,2]==uniq[j]:
            new[j]+=1
print(new)
total=sum(new)
print(total)
for i in range(0,len(Z)):
    for j in range(0,len(uniq)):
        if Z[i,2]==uniq[j]:
            Z[i,2]=format(new[j]/total,'.10f')
print(Z)
uniq1=np.unique(Z[:,3])
new1=[0]*len(uniq1)
for i in range(0,len(Z)):
    for j in range(0,len(uniq1)):
        if Z[i,3]==uniq1[j]:
            new1[j]+=1
print(new1)
print(total)
for i in range(0,len(Z)):
    for j in range(0,len(uniq1)):
        if Z[i,3]==uniq1[j]:
            Z[i,3]=format(new1[j]/total,'.10f')
Z=normalize(Z,axis=0,norm='max')
variance=[0]*len(Z[0])
for i in range(0,len(Z[0])):
    variance[i]=np.var(Z[:,i].astype(float))
avg_var=sum(variance)/len(variance)
#print(avg_var)
#print("Z=",Z)
newz=[]
deleted_rows=[]
for j in range(0,len(Z[0])):
    temp=[]
    if variance[j]<avg_var:
            deleted_rows.append(j)
            continue
    else:
        for i in range(0,len(Z)):
            temp.append(Z[i][j])
        newz.append(temp)
print(len(deleted_rows))
print("Deleted features=",deleted_rows)
print(len(newz[0]))
NZ=np.asarray(np.transpose(newz))
print("NZ=",len(NZ))
print("Length=",NZ[0])
no_chromosomes=50
chromosome=np.random.randint(0,2,size=(no_chromosomes,len(NZ[0])))
print(chromosome)
chromosome_fitness=[0]*no_chromosomes
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), random_state=0)
newtemp=[]
test=[]
for iter in range(0,2):
    for i in range(0,len(chromosome)):
        test_new=[]
        test=[]
        for j in range(0,len(chromosome[0])):
            if chromosome[i][j]==1:
                for k in range(0,len(NZ)):
                    newtemp.append(NZ[k][j])
                test.append(newtemp)
                newtemp=[]
            #print(temp)
        #test=np.array(test)
        test_new=np.asarray(np.transpose(test))
        clf.fit(test_new[0:int(len(test_new)*0.7)],out[0:int(len(out)*0.7)])
        mse=clf.score(test_new[int(len(test_new)*0.7):],out[int(len(out)*0.7):])
        chromosome_fitness[i]=mse
    for i in range(0,int(len(chromosome)-1),2):
            #print(chromosome)
        x_temp=chromosome[i][0:int(len(chromosome[i])*0.7)]
        y_temp=chromosome[i+1][int(len(chromosome[i])*0.7):]
        x_temp1=chromosome[i][int(len(chromosome[i])*0.7):]
        y_temp1=chromosome[i+1][0:int(len(chromosome[i])*0.7)]
            #print(x_temp)
            #print(y_temp)
        temp_chrom=list(x_temp)+list(y_temp)
        temp_chrom1=list(y_temp1)+list(x_temp1)
        chromosome[i]=temp_chrom
        chromosome[i+1]=temp_chrom1
    for i in range(0,len(chromosome)):
        for j in range(0,np.random.randint(1,5)):
            k=np.random.randint(0,len(chromosome[0]))
            if chromosome[i][k]==1:
                chromosome[i][k]=0
            else:
                chromosome[i][k]=1
chromosome_dict={}
for i in range(len(chromosome_fitness)):
    chromosome_dict[i]=chromosome_fitness[i]
print(chromosome_dict)
sorted_dict=sorted(chromosome_dict.items(),key=operator.itemgetter(1),reverse=True)
print(sorted_dict)
#old=[old for old in sorted_dict.keys()]
selected_chrom=sorted_dict[0][0]
i=selected_chrom
new_dict={}
k=0
for j in range(0,len(chromosome[0])):
    if chromosome[i][j]==1:
        new_dict[k]=j
        k+=1
print(new_dict)
final1_train=[]
for i in range(0,len(NZ[0])):
    test1_new=[]
    if i in new_dict.values():
        for j in range(0,len(NZ)):
            test1_new.append(NZ[j][i])
        final1_train.append(test1_new)
final_train=np.asarray(np.transpose(final1_train))
clf.fit(final_train[0:int(len(final_train)*0.7)],out[0:int(len(out)*0.7)])
print("MSE=",clf.score(final_train[int(len(final_train)*0.7):],out[int(len(out)*0.7):]))
data=open("KDDTest+.csv","r")
test=list(csv.reader(data))
counttcp=0
countudp=0
counticmp=0
for i in range(0,len(test)):
    for j in range(0,len(test[0])):
        if test[i][j]=='tcp':
            counttcp=counttcp+1
        elif test[i][j]=='icmp':
            counticmp=counticmp+1
        elif test[i][j]=='udp':
            countudp=countudp+1
#print(counticmp)
for i in range(0,len(test)):
    for j in range(0,len(test[0])):
        if test[i][j]=='tcp':
            test[i][j]=format(counttcp/(counttcp+counticmp+countudp),'.10f')
        elif test[i][j]=='icmp':
            test[i][j]=format(counticmp/(counttcp+counticmp+countudp),'.10f')
        elif test[i][j]=='udp':
            test[i][j]=format(countudp/(counttcp+counticmp+countudp),'.10f')
Z_test=np.array(test)
uniq_test=np.unique(Z_test[:,2])
new_test=[0]*len(uniq_test)
for i in range(0,len(Z_test)):
    for j in range(0,len(uniq_test)):
        if Z_test[i,2]==uniq_test[j]:
            new_test[j]+=1
#print(new)
total_test=sum(new_test)
#print(total)
for i in range(0,len(Z_test)):
    for j in range(0,len(uniq_test)):
        if Z_test[i,2]==uniq_test[j]:
            Z_test[i,2]=format(new_test[j]/total_test,'.10f')
#print(Z)
uniq1_test=np.unique(Z_test[:,3])
new1_test=[0]*len(uniq1_test)
for i in range(0,len(Z_test)):
    for j in range(0,len(uniq1_test)):
        if Z_test[i,3]==uniq1_test[j]:
            new1_test[j]+=1
#print(new1)
#print(total)
for i in range(0,len(Z_test)):
    for j in range(0,len(uniq1_test)):
        if Z_test[i,3]==uniq1_test[j]:
            Z_test[i,3]=format(new1_test[j]/total_test,'.10f')
Z_test=normalize(Z_test,axis=0,norm='max')
dict_test={}
for j in range(0,len(deleted_rows)):
    dict_test[j]=deleted_rows[j]
print("Length of Z_test=",len(Z_test))
print("Length of Z_test[0]=",len(Z_test[0]))
#newz_test=[]
pre_test=[]
for i in range(0,len(Z_test)):
    temp_test=[]
    for j in range(0,len(Z_test[0])):
        if j in dict_test.values():
            continue
        else:
            temp_test.append(Z_test[i][j])
    pre_test.append(temp_test)
'''for j in range(0,len(Z_test[0])):
    temp_test=[]
    if j in dict_test.values():
        continue
    else:
        for i in range(0,len(Z_test)):
            temp_test.append(Z_test[i][j])
        newz_test.append(temp_test)'''
#print("pre_test=",pre_test)
print("length pre_test=",len(pre_test[0]))
#print("newz_test=",newz_test)
#print("length newz_test=",len(newz_test))

'''
for i in range(0,len(pre_test[0])):
    newtemp1_test=[]
    if i in new_dict:
        for j in range(0,len(pre_test)):
            newtemp1_test.append(pre_test[j][i])
        test1_new.append(newtemp1_test)
                #print(temp)
          #test=np.array(test)'''
final1_test=[]
for i in range(0,len(pre_test[0])):
    test1_new=[]
    if i in new_dict.values():
        for j in range(0,len(pre_test)):
            test1_new.append(pre_test[j][i])
        final1_test.append(test1_new)
final_test=np.asarray(np.transpose(final1_test))
#final_test=np.asarray(np.transpose(final_test1))
print(new_dict)
print("Length of final_test",len(final_test))
#print("Final_test=",final_test)
print("Length of out_test=",len(out_test))
final_mse=clf.score(final_test,out_test)
print("Test_mse=",final_mse)
return_target=clf.predict(final_test)
print(return_target)
#mse=clf.score(final_test,targettemp_test)
    #plt.plot(test_new1,targettemp)
    #plt.show()