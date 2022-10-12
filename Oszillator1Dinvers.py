import numpy as np
import os

print("----------------")
try:
    os.add_dll_directory("C:\\Programme\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
    print("GPU Support")
except:
    print("No GPU Support")
print("----------------")

import tensorflow as tf
from tensorflow import math, random
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import time









class PINN:
    
    def __init__(self,d_l,ub,lb,t_start,t_end,sigma, n=[1,0.5,0,0]):
        
        self.d_l = d_l
        self.ub = ub
        self.lb = lb

        # ub = upper bound, lb = lower bound, d_l = Dim der Dense Layer

        self.t_start = t_start
        self.t_end = t_end

        self.traindict = {}
        
        self.m = 1
        self.omega = 1
        self.h = 1
        self.sigma = sigma

        self.hermites = {
            0: lambda x: 1,
            1: lambda x: tf.cast(2*x, dtype=tf.float64),
            2: lambda x: tf.cast(4*x**2 - 2, dtype=tf.float64),
            3: lambda x: tf.cast(8*x**3 - 12*x, dtype=tf.float64),
        }

        self.n = n/np.linalg.norm(n)
        self.factors = [tf.Variable([0,0,0,0],dtype=tf.float64)]
        self.factorlist = self.factors[0].numpy()

        # Kontrollwerte
        
        xtest = np.random.rand(100) * (ub-lb) + lb
        ttest = np.random.rand(100) * (t_end-t_start) + t_start
        Xtest, Ttest = np.meshgrid(xtest,ttest)
        xxtest = Xtest.flatten()
        tttest = Ttest.flatten()
        self.utrue, self.vtrue = self.solution(xxtest,tttest)
        self.xtest = np.stack((xxtest,tttest),axis=1)
        self.htrue = np.sqrt(self.utrue**2 + self.vtrue**2)

        self.losslist = []
        self.reallist = []
        self.imaglist = []
        self.abslist = []

        self.calliter = range(0,50000).__iter__()

        # Erstellen vom Model
        inputs = tf.keras.Input(shape=(d_l[0],))

        x = layers.Lambda(lambda x: x[:,0:1], output_shape=(1,))(inputs)
        t = layers.Lambda(lambda x: x[:,1:2], output_shape=(1,))(inputs)

        x = layers.Lambda(lambda x: (x - (ub+lb)/2) / (ub-lb) *2, output_shape=(1,))(x)
        t = layers.Lambda(lambda x: (x - (t_end+t_start)/2) / (t_end-t_start) *2, output_shape=(1,))(t)

        x = layers.Concatenate(axis=1)([x,t])

        for k in d_l[1:-1]:
            x = layers.Dense(k, activation="tanh")(x)

        outputs = layers.Dense(d_l[-1])(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PINN")


        


    def train(self, l_r, exp, epochs, s_n):

        start_time = time.time()

        opt = tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(l_r,1000,exp))
        opt2 = tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01,1000,0.7))

        # Erstellen der x und t Werte für Training
        x_train = np.random.rand(s_n) * (self.ub-self.lb) + self.lb
        t_train = np.random.rand(s_n) * (self.t_end-self.t_start) + self.t_start

        X_train = np.stack((x_train, t_train), axis=1)
        X = tf.Variable(X_train)

        u_true, v_true = self.solution(x_train,t_train)
        

        self.traindict.update({
            "X": X,
            "epochs": epochs,
            "u": u_true,
            "v": v_true
        })

        for epoch in range(epochs):


            loss, grads, factorgrads = self.calculate_loss(X,u_true,v_true)
                
            opt.apply_gradients(zip(grads, self.model.trainable_weights))
            opt2.apply_gradients(zip(factorgrads, self.factors))


         

            if epoch % 20 == 0:
                
                print("Epoche %s/%s --- Loss: %s" % (epoch,epochs,self.roundloss(float(loss))))
                self.losslist.append(loss)
                self.loss_print()


        print("--------------------------------")
        print("Adam-Optimizer beendet")
        print("--------------------------------")
        print("")
        time.sleep(1.5)
        print("")
        print("")
        print("--------------------------------")
        print("L-BFGS-B-Optimizer startet")
        print("--------------------------------")
        print("")



        weights1 = self.weights_to_arrays(tf.Variable([0.]),self.model.get_weights(),self.factors)[1]
        weights0 = weights1.numpy().astype(np.float64)



        sopt.minimize(self.soptfunc,weights0,method='L-BFGS-B',jac=True,callback=self.calling,options={'maxiter': 50000,
                                                                                            'maxfun': 50000,
                                                                                            'maxcor': 50,
                                                                                            'maxls': 50,
                                                                                            'ftol' : 1 * np.finfo(float).eps})

        
        self.traindict.update({"el_time": time.time() - start_time})




    
    def roundloss(self,x):
        return round(x, -np.int(np.floor(np.log10(abs(x))))+3)

   

    def solution(self,x,t):  # Lösung, die für Anfangsbedingungen, Randbedingungen & Kontrolle verwendet wird

        solfunr = tf.zeros(tf.shape(x),dtype=tf.float64)
        solfuni = tf.zeros(tf.shape(x),dtype=tf.float64)

        for k in range(0,len(self.n)):

            herfunx = 1/np.sqrt(2**k * np.math.factorial(k)) * \
            (self.m*self.omega/np.pi/self.h)**(0.25) * math.exp(-self.m*self.omega*x**2/2/self.h) * \
            self.hermites[k](np.sqrt(self.m*self.omega/self.h) * x)

            solfunr += math.cos(t * self.omega* (k + 0.5)) * herfunx * self.n[k]
            solfuni -= math.sin(t * self.omega* (k + 0.5)) * herfunx * self.n[k]

        solfunr += solfunr * random.normal(tf.shape(x),stddev=self.sigma,dtype=tf.float64)
        solfuni += solfuni * random.normal(tf.shape(x),stddev=self.sigma,dtype=tf.float64)

        return tf.cast(solfunr,tf.float64), tf.cast(solfuni,tf.float64)


    # Printet Realtei-, Imaginärteil- & Absolutbetragloss
    # Nimmt aber keinen Einfluss auf den Trainingsprozess
    def loss_print(self,append=True):
        ypred = self.model(self.xtest) 

        upred = ypred[:,0].numpy()
        vpred = ypred[:,1].numpy()
        hpred = np.sqrt(upred**2+vpred**2)

        real_loss = np.linalg.norm(upred-self.utrue)/np.linalg.norm(self.utrue)
        imag_loss = np.linalg.norm(vpred-self.vtrue)/np.linalg.norm(self.vtrue)
        abs_loss = np.linalg.norm(hpred-self.htrue)/np.linalg.norm(self.htrue)


        print("Realteil-Loss: %s %%" % self.roundloss(real_loss*100))
        print("Imaginärteil-Loss: %s %%" % self.roundloss(imag_loss*100))
        print("Betrag-Loss: %s %%" % self.roundloss(abs_loss*100))
        print([self.roundloss(v) for v in self.factors[0].numpy()])

        print("")

        if append:
            self.reallist.append(real_loss)
            self.imaglist.append(imag_loss)
            self.abslist.append(abs_loss)
            self.factorlist = np.vstack((self.factorlist,self.factors[0].numpy()))



    def weights_to_arrays(self,loss,grads,fac):
    
        c = []
        c.append(loss)          # Erster Eintrag der Liste
        d = []

        for k in grads:                     # grads ist eine Liste aus Tensoren
            d.append(tf.reshape(k,[-1]))        # Jeder Tensor wird geflattened

        d.append(tf.reshape(fac[0],[-1]))
        c.append(layers.Concatenate(axis=0)(d))             # Geflattende Tensoren werden zusammengesetzt

        return c        # Macht aus den Weights einen 1D Tensor



    def list_to_weights(self,c):
        d = []
        count = 0           # Zeigt an, welche Einträge schon einsortiert wurden
        
        for k in range (1,len(self.d_l)):
            d.append(np.reshape(c[count:count+self.d_l[k-1]*self.d_l[k]],(self.d_l[k-1],self.d_l[k])))      # Macht eine Matrix in Form des Dense-Layers        
            count += self.d_l[k-1]*self.d_l[k]                                                              # Einträge wurden in die Matrix einsortiert
            d.append(c[count:count+self.d_l[k]])            # Bias-Vektor
            count += self.d_l[k]                            # Einträge wurden in den Bias-Vektor einsortiert
            
        self.factors[0].assign(c[count:])

        return d




    def calling(self,xk):

        count = self.calliter.__next__()

        if count % 25 == 0:

            loss, grads, fac = self.calculate_loss(self.traindict["X"],self.traindict["u"],self.traindict["v"])

            print("L-BFGS-B Epoche %s --- Loss: %s" % (count, self.roundloss(float(loss))))

            self.losslist.append(loss)
            self.loss_print()


    
    def soptfunc2(self,weightlist):

        weights = self.list_to_weights(weightlist)             # Nimmt einen 1D-Array und bringt ihn die Form der Weights
        self.model.set_weights(weights)                        # Neue Gewichte werden verwendet

        loss, grads, fac = self.calculate_loss(self.traindict["X"],self.traindict["u"],self.traindict["v"])

        return self.weights_to_arrays(loss, grads, fac)             # Gradient hat die Form der Gewichte, muss wieder in die Form eines 1D-Tensors gebracht werden



    def soptfunc(self,weightlist):

        return [v.numpy().astype(np.float64) for v in self.soptfunc2(weightlist)]               # Konvertierung zu NumPy-Arrays




    @tf.function
    def calculate_loss(self,X,utrue,vtrue):
        

        with tf.GradientTape(persistent=True) as g:

            x = X[:,0]
            t = X[:,1]
                
            g.watch([x,t,self.factors])

            X2 = tf.stack([x,t], axis=1)
            Y = self.model(X2)

            u = tf.cast(Y[:,0], tf.float64)
            v = tf.cast(Y[:,1], tf.float64)

            #Gradienten für Diff-Glg
            u_x = g.gradient(u, x)
            u_t = g.gradient(u, t)
            u_xx = g.gradient(u_x, x)

            v_x = g.gradient(v, x)
            v_t = g.gradient(v, t)
            v_xx = g.gradient(v_x, x)


            # Diff-Glg mit separatem Real- & Imaginärteil
            fr = self.factors[0][0] * u_xx + self.factors[0][1] * x**2 * u/2 + self.h * v_t 
            fi = self.factors[0][2] * v_xx + self.factors[0][3] * x**2 * v/2 - self.h * u_t 


            # Diffgleichung-Fehler + Randbedingungen-Fehler + Anfangsbedingungen-Fehler
            
            loss = math.reduce_mean(math.square(fr)) + math.reduce_mean(math.square(fi)) + \
                (math.reduce_mean(math.squared_difference(u,utrue)) + math.reduce_mean(math.squared_difference(v,vtrue))) * 1
                    

            grads = g.gradient(loss, self.model.trainable_weights)
            factorgrads = g.gradient(loss, self.factors)
    

        return loss, grads, factorgrads


    
    def loss_plot(self):

        epochs = self.traindict["epochs"]

        x1 = np.arange(0,epochs,20)
        x2 = epochs + np.arange(0,self.calliter.__next__(),25)
        x = np.concatenate((x1,x2))


        plt.figure()
        plt.plot(x,np.stack(self.reallist))
        plt.plot(x,np.stack(self.imaglist))
        plt.plot(x,np.stack(self.abslist))

        plt.legend(["Realteil-Fehler","Imaginärteil-Fehler","Betrag-Fehler"])
        plt.yscale("log")
        plt.xlabel("Epochen")

        plt.show()

        plt.figure()
        plt.plot(x,np.stack(self.losslist))
        plt.legend(["Kostenfunktion"])
        plt.xlabel("Epochen")
        plt.yscale("log")

        plt.show()

        plt.figure()
        plt.plot(x,self.factorlist[1:,0])
        plt.plot(x,self.factorlist[1:,1])
        plt.plot(x,self.factorlist[1:,2])
        plt.plot(x,self.factorlist[1:,3])

        plt.legend(["c1","c2","c3","c4"])
        plt.xlabel("Epochen")

        plt.show()

    
    def print_time(self):

        el_time = self.traindict["el_time"]

        hours = int(np.floor(el_time/3600))
        el_time -= 3600*hours
        minutes = int(np.floor(el_time/60))
        el_time -= 60*minutes
        seconds = int(np.floor(el_time))

        print("Benötigte Zeit: %sh %smin %ss" % (hours,minutes,seconds))

    
    def fac_plot(self):

        epochs = self.traindict["epochs"]

        x1 = np.arange(0,epochs,20)
        x2 = epochs + np.arange(0,self.calliter.__next__(),25)
        x = np.concatenate((x1,x2))

        plt.figure()
        plt.plot(x,self.factorlist[1:,0])
        plt.plot(x,self.factorlist[1:,1])
        plt.plot(x,self.factorlist[1:,2])
        plt.plot(x,self.factorlist[1:,3])

        plt.legend(["1. Faktor","2. Faktor","3. Faktor","4. Faktor"])
        
        plt.show()
    



    









if __name__ == "__main__": 

    
    lb = -1
    ub = 1

    t_start = 0
    t_end = np.pi/2


    samples_number = 50000

    d_l = [2, 100, 100, 100, 100, 2]


    epochs = 10000

    learning_rate = 1e-3
    exponential_decay = 0.92

    sigma = 0.2


    Schroed_Net = PINN(d_l,ub,lb,t_start,t_end,sigma)


    Schroed_Net.model.summary()

    Schroed_Net.train(learning_rate, exponential_decay, epochs, samples_number)


    Schroed_Net.loss_print(append=False)

    Schroed_Net.print_time()

    Schroed_Net.loss_plot()




