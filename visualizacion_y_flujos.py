# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:26:57 2014

@author: diego
"""

from dingdef import crear_particulas_aleatorias, Reservorio, Caja, ReglasColision, Simulacion, plot_datos
import numpy as np
from matplotlib import pyplot as plt


frecuencia = 1.
num_total = 11
reservorio = Reservorio()
caja = Caja(12)
lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
reglas = ReglasColision(caja, reservorio)
sim = Simulacion(lista, reglas)
imprimir = 1
steps = 500
sim.run(steps, imprimir)

plot_datos(sim, num_total, frecuencia, 0)

#Flujo entre partículas


lista1 = sim.registro_velocidades["Particula1"]
lista2 = sim.t_eventos
lista3 = sim.registro_velocidades["Particula" + str(len(sim.particulas))]

print len(lista1), len(lista2)
deltaEs1 = []

for i in xrange(len(lista1)-1):
    deltaE_j = (lista1[i+1]**2 - lista1[i]**2)*0.5
    deltaEs1.append(deltaE_j)

flujo_promedio1 =  np.sum(deltaEs1)/lista2[-1]
print flujo_promedio1

deltaEs2 = []

for i in xrange(len(lista3)-1):
    deltaE_j = (lista3[i+1]**2 - lista3[i]**2)*0.5
    deltaEs2.append(deltaE_j)

flujo_promedio2 =  np.sum(deltaEs2)/lista2[-1]
print flujo_promedio2

flujo1 = [np.sum(deltaEs1[0:i]) for i in xrange(len(deltaEs1))]/np.array(lista2[-1])
flujo2 = [np.sum(deltaEs2[0:i]) for i in xrange(len(deltaEs2))]/np.array(lista2[-1])

numero_eventos = np.arange(len(flujo1))

plt.figure()
plt.plot(numero_eventos, flujo1)
plt.plot(numero_eventos, flujo2)