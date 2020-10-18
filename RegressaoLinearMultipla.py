import numpy as np
import matplotlib.pyplot as plt


x = np.array([[1,5,118],
              [1,13,132],
              [1,20,119],
              [1,28,153],
              [1,41,91],
              [1,49,118],
              [1,61,132],
              [1,62,105]])

y = np.array([[8.1],
              [6.8],
              [7.0],
              [7.4],
              [7.7],
              [7.5],
              [7.6],
              [8.0]])

# regressão //
#print(x)
xtran = np.transpose(x) #Matriz Transposta xT OU x'
#print(xtran)

xtran_x_x = np.mat(xtran) * np.mat(x) # Multiplicação Matricial xT*x ou x'*x
#print(xtran_x_x)

xinv = np.linalg.inv(xtran_x_x) # Inversão Matricial (xTx)^-1 ou (x'*x)^-1
#print(xinv)

xtran_x_y = np.mat(xtran) * np.mat(y) # Multiplicação para obter xTy ou x'*y
#print(xtran_x_y)

beta = np.mat(xinv) * np.mat(xtran_x_y) # Multiplicação de (XTX)-1(XTy) ou (x'*x)-1*(x'y) para obter os betas
#print(beta)

# regressão //

y_hat = np.dot(x,beta) # y chapeu ou com o carai do circunflexo em cima
##

media_y = 0 # y com uma barra sla um demorgan em cima = sum(y)/n sendo n = numero de informações da amostra
SQE = 0
SQT = 0
SQR = 0
r2 = 0
k = aa # quantidade de variaveis explicativas ou numero de parâmetros
n = aa # numero de informações da amostra, ou  dados contidos nas variaveis explicativas 
r2_ajustado = 0

for i in range(0,13):
    SQE = SQE + ((y[i][0] - y_hat[i][0]) **2) # SQE = SQT - SQR
    media_y = media_y + (y[i][0])


for i in range(0,13):
    SQT = SQT + ((y[i][0] - (media_y/13))**2) # tambem pode ser escrito SQT = y'*y - n*(media_y/13))**2) OU tambem SQT = SQE + SQR
    SQR = SQR + ((y_hat[i][0] - (media_y/13))**2) # SQR = (y_hat[i][0]**2) - n*(media_y/13))**2) OU beta'*x'*y - n*(media_y/13))**2)

r2 = 1 - SQE / SQT
r2_ajustado = 1 - ((n - 1)/ (n-(k+1)))*(1-r2)
 
print("SQT ",SQT)
print("SQR ", np.float(SQR))
print("SQE ", np.float(SQE))
print("R² ", np.float(r2))
print("R² Ajustado ", np.float(r2_ajustado))


#PLOTAR O GRAFICO, EM 2D E 3D
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





