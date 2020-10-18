import numpy as np
import matplotlib.pyplot as plt


x = np.array([[1.0000,  1.7400,  5.3000, 10.8000],
              [1.0000,  6.3200,  5.4200,  9.4000],
              [1.0000,  6.2200,  8.4100,  7.2000],
              [1.0000, 10.5200,  4.6300,  8.5000],
              [1.0000,  1.1900, 11.6000,  9.4000],
              [1.0000,  1.2200,  5.8500,  9.9000],
              [1.0000,  4.1000,  6.6200,  8.0000],
              [1.0000,  6.3200,  8.7200,  9.1000],
              [1.0000,  4.0800,  4.4200,  8.7000],
              [1.0000,  4.1500,  7.6000,  9.2000],
              [1.0000, 10.1500,  4.8300,  9.4000],
              [1.0000,  1.7200,  3.1200,  7.6000],
              [1.0000,  1.7000,  5.3000,  8.2000]])

y = np.array([[25.500], 
              [31.200], 
              [25.900], 
              [38.400], 
              [18.400], 
              [26.700], 
              [26.400], 
              [25.900], 
              [32.000], 
              [25.200], 
              [39.700], 
              [35.700], 
              [26.500]])


#print(x)
xtran = np.transpose(x)
#print(xtran)

xtran_x_x = np.mat(xtran) * np.mat(x)
#print(xtran_x_x)

xinv = np.linalg.inv(xtran_x_x)
#print(xinv)

xtran_x_y = np.mat(xtran) * np.mat(y)
#print(xtran_x_y)

beta = np.mat(xinv) * np.mat(xtran_x_y)
#print(beta)

ytran = np.transpose(y)
#print(ytran)


y_hat = np.dot(x,beta)


##

media_y = 0
SQE = 0
SQT = 0
SQR = 0
r2 = 0
n = 13
k = 4 # quantidade de variaveis explicativas ou xi

r2_ajustado = 0
for i in range(0,13):
    SQE = SQE + ((y[i][0] - y_hat[i][0]) **2)
    media_y = media_y + (y[i][0])


for i in range(0,13):
    SQT = SQT + ((y[i][0] - (media_y/13))**2)
    SQR = SQR + ((y_hat[i][0] - (media_y/13))**2)

r2 = 1 - SQE / SQT
r2_ajustado = 1 - ((n - 1)/ (n-(k+1)))*(1-r2)
 
print("SQT ",SQT)
print("SQR ", np.float(SQR))
print("SQE ", np.float(SQE))
print("R² ", np.float(r2))
print("R² Ajustado ", np.float(r2_ajustado))


#CASO O SENHOR QUEIRA PLOTAR O GRAFICO. NÃO SEI SE FICOU CERTO, FIZ ELE EM 2D E 3D
#for i in range(0,4):
#    plt.scatter(x[:,i],y)
#
#for i in range(0,4):
#    y_estimado = np.dot(x,beta)
#    m, b = np.polyfit(x[:,i], y, 1)
#    plt.plot(x, m*x + b,color='k')
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for i in range(0,4):
#    ax.scatter3D(x[:,i],y, cmap='Greens')
#    m, b = np.polyfit(x[:,i], y, 1)
#    plt.plot(x, m*x + b, color='k')
#
#plt.show()





