# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:41:22 2022

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


#Si se usa en colab o Jupyter descomentar el display

#Planteo Expresión de forma simbolica a ver si me la simplifica
A, kappa, k1, a, m, g, hbar, E, V0 = sp.symbols('A k_2 k_1 a m g h_{bar} E V_0', positive=True)

D = (k1 - kappa)*A*sp.exp(-k1*a)*sp.exp(-kappa*a)/(-2*kappa)
C = (k1 + kappa)*A*sp.exp(-k1*a)*sp.exp(kappa*a)/(2*kappa)

eq = (kappa*hbar**2)*(D - C) + m*g*(C + D)
#display(eq.simplify())

#Me quedo solo con la parte que tiene q ser igual a cero
eq2 = (g*m*((k1 - kappa)*sp.exp(a*k1)-(k1 + kappa)*sp.exp(a*(2*kappa + k1)) + (kappa*hbar**2)*((k1-kappa)*sp.exp(a*k1) + (k1+kappa)*sp.exp(a*(k1 + 2*kappa)))))
#display(eq2.simplify())

#Reemplazo kappa y k1
eq2 = eq2.subs(k1, sp.sqrt(2*m*E)/hbar)
eq2 = eq2.subs(kappa, sp.sqrt(2*m*(E + V0))/hbar)
#display(eq2.simplify())

#Resolución Numérica
from scipy.optimize import newton

eq2_num = sp.lambdify( (E, V0, hbar, g, m, a) , eq2, "numpy")

m=1 #au
g=np.sqrt(2) #au
a=0.5 #au
hbar = 1 #au

V0 = [0, 0.429, 0.896]

for i in range(len(V0)):
  res = newton(eq2_num, x0=0.5, args=(V0[i], hbar, g, m, a))
  print('\nPara V0=', V0[i], 'au obtengo que E=', res, 'au')
  
E = np.linspace(0, 1, 1000)

fig = plt.figure(figsize=(7,5), dpi=100)
for i in range(len(V0)):
  plt.plot(E, eq2_num(E, V0[i], hbar, g, m, a), label='$V_0:$ '+ str(V0[i]) + ' au')

plt.title('Gráfico correspondiente a ecuación resuelta anteriormente')
plt.axhline(0, ls='--', color='red', label='$f(E)=0$')
plt.legend()
plt.xlabel('Energía (au)')
plt.ylabel('$f(E)$')
plt.grid()

#Veo cuánto tiene que vales A: uso la condición de normalización
A, C, D, k1, kappa, a, x= sp.symbols('A C D k_1 k_2 a x', positive=True) 
psi1=A*sp.exp(k1*x)
psi2=C*sp.exp(kappa*x) + D*sp.exp(-kappa*x)

#Modifico C y D para tenerlos en función de A
psi2 = psi2.subs(C, (k1 - kappa)*A*sp.exp(-kappa*a)*sp.exp(-k1*a)/(-2*kappa))
psi2 = psi2.subs(D, (k1 + kappa)*A*sp.exp(kappa*a)*sp.exp(-k1*a)/(2*kappa))

#Hago que la integral de la solución desde -oo hasta 0 sea igual a 0.5 (por paridad se que se cumple esto)
int1= sp.integrate(psi1, (x, -sp.oo, -a))
int2= sp.integrate(psi2, (x, -a, 0))
eq = sp.Eq(int1 + int2, 0.5)
#display(eq)
print('\nEntonces despejando queda que:\n')
A_res= sp.solve(eq, A)[0].simplify()
#display(sp.Eq(A,A_res))
print('\nY como consecuencia obtengo que:\n')
C_res= (k1 - kappa)*A_res*sp.exp(-kappa*a)*sp.exp(-k1*a)/(-2*kappa)
D_res= (k1 + kappa)*A_res*sp.exp(kappa*a)*sp.exp(-k1*a)/(2*kappa)
#display(sp.Eq(C, C_res.simplify()))
print('')
#display(sp.Eq(D, D_res.simplify()))

#Grafico las funciones para cada Energía

E = [0.5, 0.4, 0.05]
V0=[0, 0.429, 0.826]

A_num = sp.lambdify( (k1,kappa,a) , A_res, "numpy")
C_num = sp.lambdify( (k1,kappa,a) , C_res, "numpy")
D_num = sp.lambdify((k1,kappa,a) , D_res, "numpy")

hbar=1
m=1
a= 2
k1_list= np.sqrt(2*m*E)/hbar
kappa_list= np.sqrt(2*m*(E+V0))/hbar

x1=np.linspace(-20, -a, 1000)
x2=np.linspace(-a, 0, 1000)
x3=np.linspace(0, a, 1000)
x4=np.linspace(a, 20, 1000)

colors=['orange', 'green', 'blue']

fig = plt.figure(figsize=(7,5), dpi=100)
for i in range(len(V0)):
  k1= k1_list[i]
  kappa= kappa_list[i]

  A_num_i = A_num(k1, kappa, a)
  C_num_i = C_num(k1, kappa, a)
  D_num_i = D_num(k1, kappa, a)

  psi1=A_num_i*np.exp(k1*x1)
  psi2=C_num_i*np.exp(kappa*x2) + D_num_i*np.exp(kappa*x2)
  psi3=+D_num_i*np.exp(-kappa*x3) + C_num_i*np.exp(kappa*x3)
  psi4=A_num_i*np.exp(-k1*x4) 

  plt.plot(x1,psi1, color=colors[i])
  plt.plot(x2,psi2, color=colors[i])
  plt.plot(x3,psi3, color=colors[i])
  plt.plot(x4,psi4, color=colors[i],label='$E:$ '+ str(E[i]) + ' au')

plt.title('Soluciones de Schr. independientes del tiempo')
plt.legend()
plt.xlabel('x (a.u)')
plt.ylabel('$\psi(x)$')
plt.grid()
