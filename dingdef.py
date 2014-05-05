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
        self.tiempos_eventos = [0.]
        

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
        
   def tiempo_colision_particula_oscilador(self,particula, oscilador, tol = 1e-7, n = 200., tiempo_inicial = 1e-4):
       
            x_p0 = particula.x
            x_p = particula.x
            v_p = particula.v
            x_o = oscilador.x
            a_o = abs(oscilador.a)
            f_o = oscilador.fase
            w = oscilador.omega
            eq_o = oscilador.equilibrio
            t = tiempo_inicial
            
            delta_t = abs((eq_o - x_p0)/(v_p*n))
            
#            else:
#               g = abs((x_p0 - eq_o)/a_o)
#               #El igual a 1 no es una condición física, pero sí, bastante improbable.
#               if g >= 1.:
#                    t = float('inf')
#                    return t
#               else:
#                   delta_t = abs((eq_o - x_p0)/(a_o*w*n))
#            
           
            x_p = x_p0 + t*v_p
            x_o = a_o*np.sin(w*t + f_o )  + eq_o
            
            g = sign(x_p - x_o)
            h = sign(x_p - x_o)
                
                
            while 1:
                while g == h:
                    t += delta_t
                    x_p = x_p0 + t*v_p
                    x_o = a_o*np.sin(w*t + f_o )  + eq_o
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

       if particula.etiqueta == 1:
           particula.v = self.reservorio.velocidad(1)
                
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
        self.t_eventos = [0.]


#Los siguientes diccionarios permitirán hacer el "ploteo" después
        self.registro_posiciones = dict()
        self.registro_velocidades = dict()
        self.registro_amplitudes = dict()
        self.registro_fases = dict()
        
            
        
    def actualizar_particulas(self):
        for particula in self.particulas_y_osciladores:
            self.actualizar(particula)
             
    def actualizar(self, particula):

        for tiempo in particula.tiempos_eventos:
            if tiempo in self.eventos:
                del self.eventos[tiempo]
        
        particula.tiempos_eventos = []
        
        indice = self.particulas_y_osciladores.index(particula)


        
        try:

        #Para las partículas de los extremos                
            if particula.etiqueta == -1:        
                
                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[1])    
                if dt < float('inf'):
    
                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[1])
                        particula.tiempos_eventos.append(tiempo_col)
                
                dt = self.reglas_colision.tiempo_colision_pared(particula)
                if dt < float('inf'):
                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, None)
                    particula.tiempos_eventos.append(tiempo_col)
                
                      
            elif particula.etiqueta == 1:
                
                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[-2])    
                if dt < float('inf'):
                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[-2])
                        particula.tiempos_eventos.append(tiempo_col)
                
                dt = self.reglas_colision.tiempo_colision_pared(particula)
                if dt < float('inf'):
                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, None)
                    particula.tiempos_eventos.append(tiempo_col)
            
            else:
                
                #Para el resto de partículas
    
                       
                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[indice - 1])    
                if dt < float('inf'):
                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[indice - 1])
                        particula.tiempos_eventos.append(tiempo_col)
    

                    
                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[indice + 1])    
                if dt < float('inf'):
                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[indice + 1])
                        particula.tiempos_eventos.append(tiempo_col)
                
    
    
    
        except(AttributeError):
#Esta excepción permite calcular los tiempos de colisión partícula-oscilador,"partiendo del oscilador", habrá que pensar si vale la pena

#Si el oscilador choca con su partícula del lado izquierdo.
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(self.particulas_y_osciladores[indice - 1], particula)    
            if dt < float('inf'):
                tiempo_col = self.tiempo + dt
                self.eventos[tiempo_col] = (self.particulas_y_osciladores[indice - 1], particula)
                particula.tiempos_eventos.append(tiempo_col)

#Si el oscilador choca con su partícula del lado derecho.
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(self.particulas_y_osciladores[indice + 1], particula)    
            if dt < float('inf'):
                tiempo_col = self.tiempo + dt
                self.eventos[tiempo_col] = (self.particulas_y_osciladores[indice + 1], particula)
                particula.tiempos_eventos.append(tiempo_col)
        
            
    def mover_particulas_y_osciladores(self, delta_t):
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
            t_siguiente_evento = min(self.eventos.keys())
            siguiente_evento = self.eventos[t_siguiente_evento]

#Estos datos los usaré después para plotear y calcular los flujos.            
            for j in xrange(self.longpart):
                self.registro_posiciones["Particula" + str(j+1)][i] = self.particulas[j].x
                self.registro_velocidades["Particula" + str(j+1)][i] = self.particulas[j].v
                
            for k in xrange(self.longosc):
                self.registro_amplitudes["Oscilador" + str(k+1)][i] = self.osciladores[k].a
                self.registro_fases["Oscilador" + str(k+1)][i] = self.osciladores[k].fase
                         

#Si se estrella contra la pared

            if siguiente_evento[1] is None:
                delta_t = self.reglas_colision.tiempo_colision_pared(siguiente_evento[0]) #Lo mismo que ya había hecho
                self.tiempo = t_siguiente_evento #El tiempo que había más éste delta t"
                self.mover_particulas_y_osciladores(delta_t) #Cambio posiciones del oscilador y las particulas
                self.actualizar_fases(np.float(delta_t))      #Actualizo las fases de todos los ociladores.               
                self.reglas_colision.colision_pared(siguiente_evento[0])

# Si chocan la particula y el oscilador
   
            else:
                delta_t = self.reglas_colision.tiempo_colision_particula_oscilador(siguiente_evento[0], siguiente_evento[1])
                self.tiempo = t_siguiente_evento
                self.mover_particulas_y_osciladores(delta_t)
                self.actualizar_fases(delta_t, siguiente_evento[1]) #Acualizo las fases de todos los osciladores, exceptuando el que chocó
                self.reglas_colision.colision_particula_oscilador(siguiente_evento[0], siguiente_evento[1],np.float(delta_t))


     
            self.t_eventos.append(self.tiempo) 
            
            if imprimir == 1:
                print i + 1,  self.tiempo, self.particulas_y_osciladores
 
#Esta condición es por si algo sale mal.               
            for particula in self.particulas:
                if abs(particula.x) > self.reglas_colision.caja.tamano + 1e-5:
                    raise ValueError ('Alguna particula salió de la caja' + repr(particula))



            

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
#
            for j in xrange(int(num_osciladores)):
                osx[j].append(sim.registro_amplitudes["Oscilador" +  str(j + 1) ][i]*np.sin(np.float(omega*(t - sim.t_eventos[i]) + sim.registro_fases["Oscilador" +  str(j + 1)][i])) + sim.osciladores[j].equilibrio)


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
#    np.random.seed(6)
    frecuencia = 10.
    num_total = 5
    reservorio = Reservorio()
    caja = Caja(15.)
    lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
    reglas = ReglasColision(caja, reservorio)
    sim = Simulacion(lista, reglas)
    sim.run(10,1)    
    plot_datos(sim, num_total, frecuencia, 0)
#print sim.eventos
