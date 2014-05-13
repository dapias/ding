# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:26:57 2014

@author: diego
"""

from dingdef import crear_particulas_aleatorias, Reservorio, Caja, ReglasColision, Simulacion, plot_datos
import numpy as np
from matplotlib import pyplot as plt



frecuencia = 10.
num_total = 5
reservorio = Reservorio()
caja = Caja(np.float((num_total - 1.)/2.))
lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
reglas = ReglasColision(caja, reservorio)
sim = Simulacion(lista, reglas)
imprimir = 1


sim.run(1)
#steps = 50000
#descartados = 1000

#def plot_flujos(sim):
#    
#    v_x = [[] for _ in xrange(len(sim.particulas))]
#    for j in xrange(len(sim.particulas)):    
#        for i in xrange(len(sim.t_eventos)-1):
#            v_x[j].append(sim.registro_velocidades["Particula" + str(j + 1)][i])
#    
#    
#    deltaEs = [[] for _ in xrange(len(sim.particulas))]
#    for j in xrange(len(sim.particulas)):
#        for i in xrange(len(v_x[0])-1):
#            deltaEs[j].append(((v_x[j][i+1])**2. - (v_x[j][i])**2.)*0.5)
#            
#
#    tiempo_eventos = sim.t_eventos
#
#    
#    flujo_promedio = [[] for _ in xrange(len(sim.particulas))]
#
#    for j in xrange(len(sim.particulas)):
##        flujo_promedio[j].append(np.sum(np.abs(deltaEs[j]))/tiempo_eventos[-2])
#        flujo_promedio[j].append(np.sum((deltaEs[j]))/tiempo_eventos[-2])
#        
#    print flujo_promedio
#        
#        
        
        
        
        
        
#    flujos =[]
#    
#    for j in xrange(len(sim.particulas)):
##        flujos.append( [np.sum(np.abs(deltaEs[j][n:n+i+2])) for i in xrange(len(tiempo_eventos[n+1:-1]))]/(np.array(tiempo_eventos[n+1:-1])- np.array(tiempo_eventos[n])))
#        flujos.append( [np.sum((deltaEs[j][n:n+i+2])) for i in xrange(len(tiempo_eventos[n+1:-1]))]/(np.array(tiempo_eventos[n+1:-1])- np.array(tiempo_eventos[n])))
#    promedio_flujos = []
#    
#    for j in xrange(len(sim.particulas)):
#        promedio_flujos.append(np.average(flujos[j]))
#        
#    print promedio_flujos
    
    

    
    # Vamos  a eliminar los primeros n datos
#    flujo1 = [np.sum(deltaEs1[n:n+i+2]) for i in xrange(len(tiempo_eventos[n+1:-1]))]/(np.array(tiempo_eventos[n+1:-1])- np.array(tiempo_eventos[n]))
#    flujo2 = [np.sum(deltaEs2[n:n+i+2]) for i in xrange(len(tiempo_eventos[n+1:-1]))]/(np.array(tiempo_eventos[n+1:-1])- np.array(tiempo_eventos[n]))
#    
#    #Promedio de flujos.
#    print np.average(flujo1)
#    print np.average(flujo2)
#
#    
#    numero_eventos = np.arange(len(flujos[0]))
#    numero_eventos2 = np.arange(len(deltaEs[j]))
    
#    if k == 1:
#        plt.figure()
#        for j in xrange(len(sim.particulas)):
#            plt.plot(numero_eventos[-steps/3:], flujos[j][-steps/3:],'-o')
#            plt.show()
#        
#        plt.figure()
#        for j in xrange(len(sim.particulas)):
#            plt.plot(numero_eventos2, deltaEs[j],'-o')
#            plt.show()
#        
#
#            
#   elif k == 0:
#        plt.figure()
#        for j in xrange(len(sim.particulas)):
#            plt.plot(numero_eventos, flujos[j],'-o')

    

#try:
#    while True:
#        sim.run(imprimir)
##        plot_datos(sim, num_total, frecuencia, 0)
#        
#except KeyboardInterrupt:
#    print 'interrupted!'
#    plot_flujos(sim)

    
#try:
#    sim.run(steps, imprimir)
#    plot_datos(sim, num_total, frecuencia, 0)
#    plot_flujos(sim, descartados)
#    
#except(ValueError):
#    print "Hubo un error en alguna particula"
#    plot_datos(sim, num_total, frecuencia, 0)
##    plot_flujos(sim,0,0)
    
    


    

