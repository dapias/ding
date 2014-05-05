# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:26:57 2014

@author: diego
"""

from dingdef import crear_particulas_aleatorias, Reservorio, Caja, ReglasColision, Simulacion, plot_datos
import numpy as np
from matplotlib import pyplot as plt


frecuencia =  1.
num_total = 11
reservorio = Reservorio()
caja = Caja(12.)
lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
reglas = ReglasColision(caja, reservorio)
sim = Simulacion(lista, reglas)
imprimir = 1
steps = 100

def plot_flujos(sim, plot = 0):
    
    v_particula_i = sim.registro_velocidades["Particula1"]
    tiempo_eventos = sim.t_eventos
    v_particula_d = sim.registro_velocidades["Particula" + str(len(sim.particulas))]
    
    deltaEs1 = []
    
    for i in xrange(len(v_particula_i)-1):
        deltaE_j = (v_particula_i[i+1]**2 - v_particula_i[i]**2)*0.5
        deltaEs1.append(deltaE_j)
    
    flujo_promedio1 =  np.sum(deltaEs1)/tiempo_eventos[-1]
    print flujo_promedio1
    
    deltaEs2 = []
    
    for i in xrange(len(v_particula_d)-1):
        deltaE_j = (v_particula_d[i+1]**2 - v_particula_d[i]**2)*0.5
        deltaEs2.append(deltaE_j)
    
    flujo_promedio2 =  np.sum(deltaEs2)/tiempo_eventos[-1]
    print flujo_promedio2
    
    flujo1 = [np.sum(deltaEs1[0:i]) for i in xrange(len(deltaEs1))]/np.array(tiempo_eventos[-1])
    flujo2 = [np.sum(deltaEs2[0:i]) for i in xrange(len(deltaEs2))]/np.array(tiempo_eventos[-1])
    
    numero_eventos = np.arange(len(flujo1))
    
    if plot ==1:
        plt.figure()
        plt.plot(numero_eventos[-steps/3:], flujo1[-steps/3:], '-o')
        plt.plot(numero_eventos[-steps/3:], flujo2[-steps/3:], '-o')
    elif plot ==0:
        plt.figure()
        plt.plot(numero_eventos, flujo1, '-o')
        plt.plot(numero_eventos, flujo2, '-o')
    
    
try:
    sim.run(steps, imprimir)
    plot_datos(sim, num_total, frecuencia, 0)
    plot_flujos(sim, 1)
except(ValueError):
    print "Hubo un error en alguna particula"
    plot_datos(sim, num_total, frecuencia, 0)
    plot_flujos(sim)
    
    
    

