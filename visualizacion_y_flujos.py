# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:26:57 2014

@author: diego
"""

from dingdef import crear_particulas_aleatorias, Reservorio, Caja, ReglasColision, Simulacion, plot_datos
import numpy as np


frecuencia = 10.
num_total = 5
reservorio = Reservorio()
caja = Caja(1.875)
lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
reglas = ReglasColision(caja, reservorio)
sim = Simulacion(lista, reglas)
imprimir = 1
sim.run(300, imprimir)

plot_datos(sim, num_total, frecuencia, 0)

#Flujo entre partículas


lista1 = sim.registro_velocidades["Particula1"]
lista2 = sim.t_eventos
lista3 = sim.registro_velocidades["Particula" + str(len(sim.particulas))]
deltaEs1 = []

for i in xrange(len(lista1)-1):
    deltaE_j = (lista1[i+1]**2 - lista1[i]**2)*0.5
    deltaEs1.append(deltaE_j)

flujo_promedio1 =  np.sum(deltaEs1)/lista2[-2]
print flujo_promedio1

deltaEs2 = []

for i in xrange(len(lista3)-1):
    deltaE_j = (lista3[i+1]**2 - lista3[i]**2)*0.5
    deltaEs2.append(deltaE_j)

flujo_promedio2 =  np.sum(deltaEs2)/lista2[-2]
print flujo_promedio2

gafgaf gfgaafgafg