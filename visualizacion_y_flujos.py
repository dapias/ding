# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:26:57 2014

@author: diego
"""

from dingdef import crear_particulas_aleatorias, Reservorio, Caja, ReglasColision, Simulacion, plot_datos
import numpy as np


frecuencia = 1.
num_total = 7
reservorio = Reservorio()
caja = Caja(15.)
lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
reglas = ReglasColision(caja, reservorio)
sim = Simulacion(lista, reglas)
imprimir = 0
sim.run(1000, imprimir)

plot_datos(sim, num_total, frecuencia, 1)

#Flujo entre partículas

#Flujo pared izquierda - particula
# Energía con la que llega, dividida entre el tiempo  (en este  caso, el lado izquierdo está más caliente.)


for i in xrange(len(sim.particulas_y_osciladores)):
    print sim.particulas_y_osciladores[i].tiempos_colisiones
#delta_energia 

#Flujo pared izquierda - particula
# Energía con la que llega, dividida entre el tiempo  (en este  caso, el lado izquierdo está más caliente.)

lista = sim.particulas_y_osciladores

px = [[] for _ in xrange(int(num_total)+1)]

#for i in xrange(len(sim.particulas_y_osciladores)):
#    try:
#        print sim.particulas_y_osciladores[i].velocidades_colisiones
#    except(AttributeError):
#        pass
#deltas_energia = 
#
#
#1./lista[0].tiempos_colisiones[-1]*(sum())

px[0].append(0)
for tiempo in sim.particulas_y_osciladores[0].tiempos_colisiones:
    if not tiempo in sim.particulas_y_osciladores[1].tiempos_colisiones:
        px[0].append(sim.particulas_y_osciladores[0].tiempos_colisiones.index(tiempo))


for i in xrange(len(sim.particulas_y_osciladores) - 1):
    for tiempo in sim.particulas_y_osciladores[i].tiempos_colisiones:
        if tiempo in sim.particulas_y_osciladores[i + 1].tiempos_colisiones:
            px[i+1].append(sim.particulas_y_osciladores[i].tiempos_colisiones.index(tiempo))

px[-1].append(0)    
for tiempo in sim.particulas_y_osciladores[-1].tiempos_colisiones:
    if not tiempo in sim.particulas_y_osciladores[-2].tiempos_colisiones:
        px[-1].append(sim.particulas_y_osciladores[-1].tiempos_colisiones.index(tiempo))
        

#Flujo pared izquierda - particula
# Energía con la que llega, dividida entre el tiempo  (en este  caso, el lado izquierdo está más caliente.)
flujos = []
for j in xrange(int((num_total)/2) + 1):
    flujo = 1./lista[2*j].tiempos_colisiones[px[2*j][-1]]*np.sum([lista[2*j].velocidades_colisiones[px[2*j][i + 1]]**2*0.5 - lista[2*j].velocidades_colisiones[px[2*j][i + 1] - 1]**2*0.5 for i in xrange(len(px[2*j])-1)])
    flujos.append(flujo)
#print flujoflujo = 1./lista[0].tiempos_colisiones[px[0][-1]]*np.sum([lista[0].velocidades_colisiones[px[0][i + 1]]**2*0.5 - lista[0].velocidades_colisiones[px[0][i + 1] - 1]**2*0.5 for i in xrange(len(px[0])-1)])

print flujos
