# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:38:56 2014

@author: diego
"""

import numpy as np
import random
from matplotlib import pyplot as plt


def sign(number):return cmp(number,0)

class Oscilador(object):
    """clase Oscilador que recibe amplitud, fase, frecuencia y posición inicial"""

    def __init__(self, equilibrio, amplitud, fase, omega):
        self.a = amplitud
        self.fase = fase
        self.omega = omega
        self.equilibrio = equilibrio
        self.x = equilibrio
        self.tiempos_colisiones = [0.]
#        self.posiciones = [self.a*np.sin(self.fase) + self.equilibrio]
#        self.tiempos_eventos = [0.]
        

    def __repr__(self):
        return "Oscilador(%s,%s,%s,%s)"%(self.x,self.a,self.fase,self.omega)
        

    def movimiento(self,delta_t):
        self.x = self.a* np.sin(self.omega*delta_t + self.fase) + self.equilibrio
        
       

class Particula_libre(object):
    
    """ Recibe posición y velocidad inicial"""
    def __init__(self,x,v,etiqueta = 0):            
       self.x = x
       self.v = v
       self.tiempos_eventos = []
       self.tiempos_colisiones = [0.]
       self.etiqueta = etiqueta
       self.velocidades_colisiones = [v]
    def __repr__(self):
        return "Partícula(%s,%s)"%(self.x,self.v)
        
    def movimiento(self,delta_t):
        self.x += delta_t * self.v
    
class Reservorio(object):
    
    def __init__(self, temperatura = 2.5, deltaT = 1, sentido = 0):
        self.sentido = sentido
        self.temperatura = temperatura
        self.deltaT = deltaT
        
    def velocidad(self, j = 0):
        i = np.random.random()
        if self.sentido == 0:
            vel = (np.sqrt(-np.log(1-i)*2*self.temperatura))
        if self.sentido == 1 or j==1:
            vel = -(np.sqrt(-np.log(1-i)*2*(self.temperatura-self.deltaT)))
        return vel
        
class Caja(object):
    """Es la longitud de la mitad de la caja, estoy suponiendo 
    que la extensión de la caja es de [-tamano,tamano]"""
    def __init__(self,tamano = 100):
        self.tamano = tamano
        
class ReglasColision(object):
   
   def __init__(self, caja, reservorio):
        self.caja = caja 
        self.reservorio = reservorio 
        
   def tiempo_colision_particula_oscilador(self,particula_i, oscilador_j, tol = 1e-6, n = 100., tiempo_inicial = 1e-3):
       
            x_p0 = particula_i.x
            x_p = particula_i.x
            v_p = particula_i.v
            x_o = oscilador_j.x
            a_o = abs(oscilador_j.a)
            f_o = oscilador_j.fase
            w = oscilador_j.omega
            eq_o = oscilador_j.equilibrio
            
            t = tiempo_inicial
            
            if v_p < 0:
                delta_t = abs((-a_o + eq_o - x_p0)/(v_p*n))
            elif v_p >0:
                delta_t = abs((a_o + eq_o - x_p0)/(v_p*n))
            else:
               g = abs((x_p0 - eq_o)/a_o)
               #El igual a 1 no es una condición física, pero sí, bastante improbable.
               if g >= 1.:
                    t = float('inf')
                    return t
               else:
                   delta_t = abs((eq_o - x_p0)/(a_o*w*n))
            
           
            x_p = x_p0 + t*v_p
            x_o = a_o*np.sin(w*t + f_o )  + eq_o
            
            g = sign(x_p - x_o)
            h = sign(x_p - x_o)
                
                
            while 1:
                while g == h:
                    t += delta_t
                    x_p = x_p0 + t*v_p
                    x_o = a_o*np.sin(w*t + f_o )  + eq_o
#                        print t, x_p, x_o
                    h = sign(x_p - x_o)
                    
                    if abs(x_p) > self.caja.tamano:
                        t = float('inf')
                        return t

                if abs(x_p - x_o) < tol:
                    break
                
                t = t - delta_t
                delta_t = delta_t*0.5
                h = -h  
            return t    
            
   def colision_particula_oscilador(self, particula_i,oscilador_j,delta_t):
   # Actualiza velocidades, amplitudes y fases.
      v_vieja = particula_i.v
      a_vieja = oscilador_j.a
      f_vieja = oscilador_j.fase
      w = oscilador_j.omega
   
      h = v_vieja**2 + (a_vieja**2*oscilador_j.omega**2)/2.*(1. - np.cos(2.*(oscilador_j.omega*delta_t + f_vieja)))
      
#      particula_i.v = (a_vieja*np.cos(oscilador_j.omega*delta_t + f_vieja))

      
      #La amplitud siempre va a ser positiva
      oscilador_j.a = np.sqrt(h)/oscilador_j.omega
   
      if oscilador_j.x < oscilador_j.equilibrio:
          oscilador_j.fase = -abs(np.arccos(v_vieja/np.sqrt(h)))
      else:
          oscilador_j.fase = abs(np.arccos(v_vieja/np.sqrt(h)))
      
      if abs(oscilador_j.fase - np.pi) < 1e-4 and oscilador_j.x < oscilador_j.equilibrio :
          oscilador_j.fase = -abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))
          
      elif abs(oscilador_j.fase - np.pi) < 1e-4 and oscilador_j.x > oscilador_j.equilibrio :
          oscilador_j.fase = abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))
      
      elif abs(oscilador_j.fase) < 1e-4 and oscilador_j.x > oscilador_j.equilibrio :
          oscilador_j.fase = abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))
          
     
      elif abs(oscilador_j.fase) < 1e-4 and oscilador_j.x < oscilador_j.equilibrio :
          oscilador_j.fase = -abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))
         
      particula_i.v = v_vieja + a_vieja*w*np.cos(w*delta_t + f_vieja)  - oscilador_j.a*w*np.cos(oscilador_j.fase)
       

   
  
   def colision_pared(self, particula):
       
       if particula.etiqueta == -1:
           particula.v = self.reservorio.velocidad(0)
#           particula.v = +0.7

       if particula.etiqueta == 1:
           particula.v = self.reservorio.velocidad(1)
#           particula.v = -particula.v
#           particula.v = -1

    
                
   def tiempo_colision_pared(self, particula):
       dt = float('inf')
       if particula.etiqueta == -1:
           if particula.v < 0:
               dt = (self.caja.tamano + particula.x)/-particula.v 
       elif particula.etiqueta == 1:
           if particula.v > 0:
               dt = (self.caja.tamano - particula.x)/particula.v
               
       return dt
         

class Simulacion(object):
    
    def __init__(self, particulas_y_osciladores, reglas_colision=None):
        
        self.particulas_y_osciladores = particulas_y_osciladores
        self.particulas = [particula for particula in self.particulas_y_osciladores if isinstance(particula,Particula_libre)]
        self.osciladores = [oscilador for oscilador in self.particulas_y_osciladores if isinstance(oscilador, Oscilador)]
        self.longpart = len(self.particulas)
        self.longosc = len(self.osciladores)

        if reglas_colision is None:
            reglas_colision = ReglasColision()
        self.reglas_colision = reglas_colision
        self.eventos = dict()
        self.tiempo = 0
        
        
#        self.actualizar_particulas()
        
        self.t_eventos = [0.]

        self.registro_posiciones = dict()
        self.registro_velocidades = dict()
        self.registro_amplitudes = dict()
        self.registro_fases = dict()
        
            
        
    def actualizar_particulas(self):
        for particula in self.particulas_y_osciladores:
            if isinstance(particula,Particula_libre):
                self.actualizar(particula)
             
    def actualizar(self, particula):

#        print "particula.tiempos_eventos:", particula.tiempos_eventos
        
#        print "Eventos:", self.eventos
        
        for tiempo in particula.tiempos_eventos:
            if tiempo in self.eventos:
                del self.eventos[tiempo]
        
#        print "Eventos:", self.eventos

#        self.eventos = {}

        particula.tiempos_eventos = []
        
        indice = self.particulas_y_osciladores.index(particula)
        


        
        #Para las partículas de los extremos
        
        if particula.etiqueta == -1:        
            
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[1])    
            
#            print "DeltaTiempo colisión con oscilador", dt
            
            if dt < float('inf'):

                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[1])
                    particula.tiempos_eventos.append(tiempo_col)
            
            dt = self.reglas_colision.tiempo_colision_pared(particula)
#            print "DeltaTiempo colisión con pared", dt
            
            if dt < float('inf'):
                tiempo_col = self.tiempo + dt
                self.eventos[tiempo_col] = (particula, None)
                particula.tiempos_eventos.append(tiempo_col)
            
            
#                self.particulas_y_osciladores[0].tiempos_colisiones.append(tiempo_col)
                  
        elif particula.etiqueta == 1:
            
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[-2])    
            
#            print "DeltaTiempo colisión con oscilador", dt
            
            if dt < float('inf'):

                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[-2])
                    particula.tiempos_eventos.append(tiempo_col)
            
            dt = self.reglas_colision.tiempo_colision_pared(particula)
#            print "DeltaTiempo colisión con pared", dt
            
            if dt < float('inf'):
                tiempo_col = self.tiempo + dt
                self.eventos[tiempo_col] = (particula, None)
                particula.tiempos_eventos.append(tiempo_col)
            
        
        else:
    
#Sólo consideramos las partículas
                        
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[indice - 1])    
            
#            print "DeltaTiempo colisión con el oscilador del lado izquierdo", dt
            
            if dt < float('inf'):

                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[indice - 1])
                    particula.tiempos_eventos.append(tiempo_col)
                    
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[indice + 1])    
            
#            print "DeltaTiempo colisión con el oscilador del lado derecho", dt
            
            if dt < float('inf'):

                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[indice + 1])
                    particula.tiempos_eventos.append(tiempo_col)
                
         
               
                
                
                
            
    def mover_particulas_y_osciladores(self, delta_t):
        
        #Para partículas.
        for particula in self.particulas_y_osciladores:
                particula.movimiento(np.float(delta_t))

                
    def actualizar_fases(self, delta_t, oscilador_prohibido = Oscilador(0,0,0,0)):
        for oscilador in self.particulas_y_osciladores:
            if oscilador != oscilador_prohibido:
                try:
                    oscilador.fase += delta_t*oscilador.omega
                except(AttributeError):
                    pass
        

    def run(self, steps=10, imprimir = 0 ):
        
        
        self.registro_posiciones = {"Particula" + str(i + 1) : np.ones(steps) for i in range(int(self.longpart))}
        self.registro_velocidades = {"Particula" + str(i + 1) : np.ones(steps) for i in range(int(self.longpart))}
        self.registro_amplitudes = {"Oscilador" + str(i + 1) : np.ones(steps) for i in range(int(self.longosc))}
        self.registro_fases = {"Oscilador" + str(i + 1) : np.ones(steps) for i in range(int(self.longosc))}
        
        for i in xrange(steps):
            
            self.actualizar_particulas()

#            print "Eventos:", self.eventos
               
            t_siguiente_evento = min(self.eventos.keys())

            siguiente_evento = self.eventos[t_siguiente_evento]
            

            
            for j in xrange(self.longpart):
                self.registro_posiciones["Particula" + str(j+1)][i] = self.particulas[j].x
                self.registro_velocidades["Particula" + str(j+1)][i] = self.particulas[j].v
                
            for k in xrange(self.longosc):
                self.registro_amplitudes["Oscilador" + str(k+1)][i] = self.osciladores[k].a
                self.registro_fases["Oscilador" + str(k+1)][i] = self.osciladores[k].fase
                         

            #Tiempo de la última colisión:
                
            

            #Si se estrella contra la pared

            if siguiente_evento[1] is None:
                delta_t = self.reglas_colision.tiempo_colision_pared(siguiente_evento[0]) #Lo mismo que ya había hecho
                self.tiempo = t_siguiente_evento #El tiempo que había más éste delta t"
                self.mover_particulas_y_osciladores(delta_t) #Cambio posiciones del oscilador y las particulas
                
# Aquí debería poner algo para actualizr la fase de todos los osciladores que no chocaron.                
                self.actualizar_fases(np.float(delta_t))                    
                self.reglas_colision.colision_pared(siguiente_evento[0])
#                print self.particulas_y_osciladores[0].etiqueta
#                self.actualizar(siguiente_evento[0])

#Para poder calcular los flujos.

                self.particulas_y_osciladores[self.particulas_y_osciladores.index(siguiente_evento[0])].tiempos_colisiones.append(t_siguiente_evento)
                self.particulas_y_osciladores[self.particulas_y_osciladores.index(siguiente_evento[0])].velocidades_colisiones.append(siguiente_evento[0].v)
                

# Si chocan la particula y el oscilador
   
            else:
                delta_t = self.reglas_colision.tiempo_colision_particula_oscilador(siguiente_evento[0], siguiente_evento[1])
                self.tiempo = t_siguiente_evento
                self.mover_particulas_y_osciladores(delta_t)


# Aquí debería poner algo para actualizr la fase de todos los osciladores que no chocaron.                

                self.actualizar_fases(delta_t, siguiente_evento[1])
                self.reglas_colision.colision_particula_oscilador(siguiente_evento[0], siguiente_evento[1],np.float(delta_t))
                
#Para poder calcular los flujos                
                self.particulas_y_osciladores[self.particulas_y_osciladores.index(siguiente_evento[1])].tiempos_colisiones.append(t_siguiente_evento)
                self.particulas_y_osciladores[self.particulas_y_osciladores.index(siguiente_evento[0])].tiempos_colisiones.append(t_siguiente_evento)
                self.particulas_y_osciladores[self.particulas_y_osciladores.index(siguiente_evento[0])].velocidades_colisiones.append(siguiente_evento[0].v)
#                self.actualizar_particulas() 
        
        
            self.t_eventos.append(self.tiempo) 
            
            if imprimir == 1:
                print i + 1,  self.tiempo, self.particulas_y_osciladores
            

def crear_particulas_aleatorias(tamano_caja, num_particulas_y_osciladores, omega, reservorio):
    #    np.random.seed(seed)
    particulas_y_osciladores = []


    for i in xrange(num_particulas_y_osciladores):
        #Si son partículas
        if i % 2 == 0:
            x = -tamano_caja + (2.*tamano_caja)*(i+1.)/(num_particulas_y_osciladores+1.)
           
            if i == 0:
                v = (random.choice([1.,-1.]))*reservorio.velocidad()
                nueva_particula = Particula_libre(x, v, -1)
            elif i == num_particulas_y_osciladores-1:
#                reservorio.sentido == 1
                v = (random.choice([1.,-1.]))*reservorio.velocidad(1)
                nueva_particula = Particula_libre(x, v, 1)
            else:
                v = np.random.uniform(-reservorio.velocidad(1),reservorio.velocidad())
                nueva_particula = Particula_libre(x, v)
                
        else:
            x_eq = -tamano_caja + (2.*tamano_caja)*(i+1.)/(num_particulas_y_osciladores+1.)
            A = np.random.uniform(tamano_caja/(2.*(num_particulas_y_osciladores+1.)))
            Fase = np.random.uniform(0,np.pi)
            nueva_particula = Oscilador(x_eq,A,Fase,omega)
            
            
        particulas_y_osciladores.append(nueva_particula)
        
    print particulas_y_osciladores

    return particulas_y_osciladores

def plot_datos(sim, total_particulas, omega, puntos = 1):
    tiempo = []
    num_particulas = (total_particulas - 1) * 0.5 + 1
#    num_particulas = 1
    num_osciladores = (total_particulas - 1) * 0.5
    px = [[] for _ in xrange(int(num_particulas))]
    osx = [[] for _ in xrange(int(num_osciladores))]
    
    if puntos == 1:
        tiempo_exacto = []
        px_exacto = [[] for _ in xrange(int(num_particulas))]
        osx_exacto = [[] for _ in xrange(int(num_osciladores))]
        
        for i in xrange(len(sim.t_eventos)-1):
            for j in xrange(int(num_particulas)):
                px_exacto[j].append(sim.registro_posiciones["Particula" + str(j + 1)][i])
            for j in xrange(int(num_osciladores)):
                osx_exacto[j].append(sim.registro_amplitudes["Oscilador" +  str(j + 1) ][i]*np.sin(np.float(sim.registro_fases["Oscilador" +  str(j + 1)][i])) + sim.osciladores[j].equilibrio)
                
            tiempo_exacto.append(sim.t_eventos[i])
      
        for j in xrange(int(num_particulas)):
            plt.plot(tiempo_exacto, px_exacto[j],'o')
        for j in xrange(int(num_osciladores)):
            plt.plot(tiempo_exacto, osx_exacto[j], 'o')

    t = 0        
    dt = 0.005
    


    for i in xrange(len(sim.t_eventos) - 1):
    
        
        while  t < float(sim.t_eventos[i + 1]):
            
            for j in xrange(int(num_particulas)):
                px[j].append(sim.registro_posiciones["Particula" + str(j + 1)][i] + (t - sim.t_eventos[i]) * sim.registro_velocidades["Particula" + str(j + 1)][i])
#            px.append(sim.registro_posiciones["Particula1"][i] + (t - sim.t_eventos[i]) * sim.registro_velocidades["Particula1"][i])
            for j in xrange(int(num_osciladores)):
                osx[j].append(sim.registro_amplitudes["Oscilador" +  str(j + 1) ][i]*np.sin(np.float(omega*(t - sim.t_eventos[i]) + sim.registro_fases["Oscilador" +  str(j + 1)][i])) + sim.osciladores[j].equilibrio)
                
#                osx[j].append(sim.registro_posiciones["Particula" + str(j + 1)][i] + (t - sim.t_eventos[i]) * sim.registro_velocidades["Particula" + str(j + 1)][i])
#            osx.append(sim.registro_amplitudes["Oscilador1"][i]*np.sin(np.float(omega*(t - sim.t_eventos[i]) + sim.registro_fases["Oscilador1"][i])) + sim.osciladores[0].equilibrio)
                

            tiempo.append(t)
            t += dt
        


                

    
 
    for j in xrange(int(num_particulas)):
        plt.plot(tiempo, px[j])
    for j in xrange(int(num_osciladores)):
        plt.plot(tiempo, osx[j])
    
    #plt.axis([0,25,-2,2])
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.show()


if __name__ == '__main__':
#    np.random.seed(2)
    frecuencia = 4
    num_total = 7
    reservorio = Reservorio()
    caja = Caja(15.)
    lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
    reglas = ReglasColision(caja, reservorio)
    sim = Simulacion(lista, reglas)
    sim.run(10,1)    
    plot_datos(sim, num_total, frecuencia, 1)
#print sim.eventos
