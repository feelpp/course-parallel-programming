import pandas as pd
import csv
import json
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')
#print(df.head())
#print(df) 
x=df['DimMatrix']
nb=len(x)
y1=df['ModeSerial']
y2=df['ModeGpuHipAMD']
y3=df['pthread']

for i in range(0,nb):
    sc=1000.0
    y1[i]=y1[i]/sc
    y2[i]=y2[i]/sc
    y3[i]=y3[i]/sc

figure1 = plt.figure()
plt.plot(x,y1,label='CPU-Serial',marker='o', linestyle='--', color='r')  
plt.plot(x,y2,label='HIP-AMD-GPU',marker='o', linestyle='--', color='g')
plt.plot(x,y3,label='pthread 9',marker='o', linestyle='--', color='b')
plt.grid()
plt.xlabel('Dim Matrix')
#plt.ylabel('Time in microsecond')
plt.ylabel('Time in millisecond')
plt.title('Cholesky factorisation resolution')
plt.xticks(rotation=90)
plt.legend()
plt.show()
plt.savefig("Data.jpg")


ratio=[]
for i in range(0,nb):
    ratio.append(y1[i]/y2[i])

figure2 = plt.figure()
plt.plot(x,ratio,label='Ratio',marker='o', linestyle='--', color='r')  
plt.grid()
plt.xlabel('Dim Matrix')
plt.ylabel('Ratio')
plt.title('Cholesky factorisation resolution')
plt.xticks(rotation=90)
plt.legend()
plt.show()
plt.savefig("DataRatio.jpg")




