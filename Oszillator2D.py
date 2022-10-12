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
from tensorflow import math
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import time









class PINN:
    
    def __init__(self,d_l,ub,lb,t_start,t_end, n=[[0.4,2,0.5,1],[2,1,3,0.5],[0.7,0.5,0.9,0.1],[0,0,2,0.4]]):
        
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

        self.hermites = {
            0: lambda x: 1,
            1: lambda x: tf.cast(2*x, dtype=tf.float64),
            2: lambda x: tf.cast(4*x**2 - 2, dtype=tf.float64),
            3: lambda x: tf.cast(8*x**3 - 12*x, dtype=tf.float64),
        }

        self.n = n/np.linalg.norm(n)

        # Kontrollwerte
        
        xtest = np.random.rand(100) * (ub-lb) + lb
        ttest = np.random.rand(100) * (t_end-t_start) + t_start
        Xtest, Ttest = np.meshgrid(xtest,ttest)
        xxtest = Xtest.flatten()
        tttest = Ttest.flatten()
        self.utrue, self.vtrue = self.solution(xxtest,xxtest,tttest)
        self.xtest = np.stack((xxtest,xxtest,tttest),axis=1)
        self.htrue = np.sqrt(self.utrue**2 + self.vtrue**2)

        self.losslist = []
        self.reallist = []
        self.imaglist = []
        self.abslist = []

        self.calliter = range(0,50000).__iter__()

        # Erstellen vom Model
        inputs = tf.keras.Input(shape=(d_l[0],))

        x = layers.Lambda(lambda x: x[:,0:1], output_shape=(1,))(inputs)
        y = layers.Lambda(lambda x: x[:,1:2], output_shape=(1,))(inputs)
        t = layers.Lambda(lambda x: x[:,2:3], output_shape=(1,))(inputs)

        x = layers.Lambda(lambda x: (x - (ub+lb)/2) / (ub-lb) *2, output_shape=(1,))(x)
        y = layers.Lambda(lambda x: (x - (ub+lb)/2) / (ub-lb) *2, output_shape=(1,))(y)
        t = layers.Lambda(lambda x: (x - (t_end+t_start)/2) / (t_end-t_start) *2, output_shape=(1,))(t)

        x = layers.Concatenate(axis=1)([x,y,t])

        for k in d_l[1:-1]:
            x = layers.Dense(k, activation="tanh")(x)

        outputs = layers.Dense(d_l[-1])(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PINN")


        


    def train(self, l_r, exp, epochs, i_n, b_n, s_n):

        start_time = time.time()

        opt = optimizers.Adam(learning_rate = optimizers.schedules.ExponentialDecay(l_r,1000,exp))

        # Erstellen der x und t Werte für Training
        x_train = np.random.rand(s_n) * (self.ub-self.lb) + self.lb
        y_train = np.random.rand(s_n) * (self.ub-self.lb) + self.lb
        t_train = np.random.rand(s_n) * (self.t_end-self.t_start) + self.t_start

        X_train = np.stack((x_train, y_train, t_train), axis=1)
        X = tf.Variable(X_train)

        x0 = np.random.rand(i_n) * (self.ub-self.lb) + self.lb          # x0 ... x Werte bei t=0
        y0 = np.random.rand(i_n) * (self.ub-self.lb) + self.lb          # y0 ... y Werte bei t=0

        u0_true, v0_true = self.solution(x0, y0, tf.zeros(tf.shape(x0),dtype=tf.float64))

        tb = np.random.rand(b_n) * (self.t_end - self.t_start) + self.t_start
        xb = tf.constant(np.random.rand(b_n) * (self.ub-self.lb) + self.lb)

        ulbx_true, vlbx_true = self.solution(tf.constant(np.ones(b_n))*self.lb, xb, tb)
        uubx_true, vubx_true = self.solution(tf.constant(np.ones(b_n))*self.ub, xb, tb)
        ulby_true, vlby_true = self.solution(xb, tf.constant(np.ones(b_n))*self.lb, tb)
        uuby_true, vuby_true = self.solution(xb, tf.constant(np.ones(b_n))*self.ub, tb)

        self.traindict.update({
            "X": X,
            "x0": x0,
            "y0": y0,
            "tb": tb,
            "xb": xb,
            "epochs": epochs,
            "b_n": b_n,
            "u0_true": u0_true,
            "v0_true": v0_true,
            "ulbx_true": ulbx_true,
            "vlbx_true": vlbx_true,
            "uubx_true": uubx_true,
            "vubx_true": vubx_true,
            "ulby_true": ulby_true,
            "vlby_true": vlby_true,
            "uuby_true": uuby_true,
            "vuby_true": vuby_true
        })
    
        for epoch in range(epochs):


            loss, grads = self.calculate_loss(X,x0,y0,xb,tb,u0_true,v0_true,
                                                ulbx_true,vlbx_true,uubx_true,vubx_true,
                                                ulby_true,vlby_true,uuby_true,vuby_true)
                
            opt.apply_gradients(zip(grads, self.model.trainable_weights))



         

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



        weights1 = self.weights_to_arrays(tf.Variable([0.]),self.model.get_weights())[1]
        weights0 = weights1.numpy().astype(np.float64)



        sopt.minimize(self.soptfunc,weights0,method='L-BFGS-B',jac=True,callback=self.calling,options={'maxiter': 50000,
                                                                                            'maxfun': 50000,
                                                                                            'maxcor': 50,
                                                                                            'maxls': 50,
                                                                                            'ftol' : 1 * np.finfo(float).eps})

        
        self.traindict.update({"el_time": time.time() - start_time})




    
    def roundloss(self,x):
        return round(x, -np.int(np.floor(np.log10(abs(x))))+3)

   

    def solution(self,x,y,t):  # Lösung, die für Anfangsbedingungen, Randbedingungen & Kontrolle verwendet wird

        solfunr = tf.zeros(tf.shape(x),dtype=tf.float64)
        solfuni = tf.zeros(tf.shape(x),dtype=tf.float64)

        for k in range(0,len(self.n)):
            for l in range(0,len(self.n)):

                herfunx = 1/np.sqrt(2**k * np.math.factorial(k)) * \
                (self.m*self.omega/np.pi/self.h)**(0.25) * math.exp(-self.m*self.omega*x**2/2/self.h) * \
                self.hermites[k](np.sqrt(self.m*self.omega/self.h) * x)

                herfuny = 1/np.sqrt(2**k * np.math.factorial(l)) * \
                (self.m*self.omega/np.pi/self.h)**(0.25) * math.exp(-self.m*self.omega*y**2/2/self.h) * \
                self.hermites[l](np.sqrt(self.m*self.omega/self.h) * y)

                solfunr += math.cos(t * self.omega* (k + l + 1)) * herfunx * herfuny * self.n[k][l]
                solfuni -= math.sin(t * self.omega* (k + l + 1)) * herfunx * herfuny * self.n[k][l]


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

        print("")

        if append:
            self.reallist.append(real_loss)
            self.imaglist.append(imag_loss)
            self.abslist.append(abs_loss)



    def weights_to_arrays(self,loss,grads):
    
        c = []
        c.append(loss)          # Erster Eintrag der Liste
        d = []

        for k in grads:                     # grads ist eine Liste aus Tensoren
            d.append(tf.reshape(k,[-1]))        # Jeder Tensor wird geflattened

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
            
        return d




    def calling(self,xk):

        count = self.calliter.__next__()

        if count % 25 == 0:

            loss, grads = self.calculate_loss(self.traindict["X"],self.traindict["x0"],self.traindict["y0"],self.traindict["xb"],self.traindict["tb"],
                                                self.traindict["u0_true"],self.traindict["v0_true"],self.traindict["ulbx_true"],self.traindict["vlbx_true"],
                                                self.traindict["uubx_true"],self.traindict["vubx_true"],self.traindict["ulby_true"],self.traindict["vlby_true"],
                                                self.traindict["uuby_true"],self.traindict["vuby_true"])

            print("L-BFGS-B Epoche %s --- Loss: %s" % (count, self.roundloss(float(loss))))

            self.losslist.append(loss)
            self.loss_print()


    
    def soptfunc2(self,weightlist):

        weights = self.list_to_weights(weightlist)             # Nimmt einen 1D-Array und bringt ihn die Form der Weights
        self.model.set_weights(weights)                        # Neue Gewichte werden verwendet

        loss, grads = self.calculate_loss(self.traindict["X"],self.traindict["x0"],self.traindict["y0"],self.traindict["xb"],self.traindict["tb"],
                                            self.traindict["u0_true"],self.traindict["v0_true"],self.traindict["ulbx_true"],self.traindict["vlbx_true"],
                                            self.traindict["uubx_true"],self.traindict["vubx_true"],self.traindict["ulby_true"],self.traindict["vlby_true"],
                                            self.traindict["uuby_true"],self.traindict["vuby_true"])

        return self.weights_to_arrays(loss, grads)             # Gradient hat die Form der Gewichte, muss wieder in die Form eines 1D-Tensors gebracht werden



    def soptfunc(self,weightlist):

        return [v.numpy().astype(np.float64) for v in self.soptfunc2(weightlist)]               # Konvertierung zu NumPy-Arrays




    @tf.function
    def calculate_loss(self,X,x0,y0,xb,tb,u0_true,v0_true,ulbx_true,vlbx_true,uubx_true,vubx_true,ulby_true,vlby_true,uuby_true,vuby_true):
        
        b_n = self.traindict["b_n"]

        with tf.GradientTape(persistent=True) as g:

            x = X[:,0]
            y = X[:,1]
            t = X[:,2]
                
            g.watch([x,y,t])

            X2 = tf.stack([x,y,t], axis=1)
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

            u_y = g.gradient(u, y)
            u_yy = g.gradient(u_y, y)

            v_y = g.gradient(v, y)
            v_yy = g.gradient(v_y, y)

            # Diff-Glg mit separatem Real- & Imaginärteil
            fr = -self.h**2 /2/self.m * (u_xx + u_yy) + self.m *self.omega**2 * (x**2 + y**2) *u/2 + self.h * v_t 
            fi = -self.h**2 /2/self.m * (v_xx + v_yy) + self.m *self.omega**2 * (x**2 + y**2) *v/2 - self.h * u_t 
                

            
            # Randbedingungen
            Xlbx2 = tf.stack([tf.constant(np.ones(b_n))*self.lb, xb, tb], axis=1)
            Xubx2 = tf.stack([tf.constant(np.ones(b_n))*self.ub, xb, tb], axis=1)
            Xlby2 = tf.stack([xb, tf.constant(np.ones(b_n))*self.lb, tb], axis=1)
            Xuby2 = tf.stack([xb, tf.constant(np.ones(b_n))*self.ub, tb], axis=1)

            Ylbx = self.model(Xlbx2)
            Yubx = self.model(Xubx2)
            Ylby = self.model(Xlby2)
            Yuby = self.model(Xuby2)

            ulbx = tf.cast(Ylbx[:,0], tf.float64)
            vlbx = tf.cast(Ylbx[:,1], tf.float64)
            uubx = tf.cast(Yubx[:,0], tf.float64)
            vubx = tf.cast(Yubx[:,1], tf.float64)
            ulby = tf.cast(Ylby[:,0], tf.float64)
            vlby = tf.cast(Ylby[:,1], tf.float64)
            uuby = tf.cast(Yuby[:,0], tf.float64)
            vuby = tf.cast(Yuby[:,1], tf.float64)


            
            # Anfangsbedingungen
            X02 = tf.stack([x0, y0, tf.zeros(tf.shape(x0),dtype=tf.float64)], axis=1)
            Y0 = self.model(X02)

            u0 = tf.cast(Y0[:,0], tf.float64)
            v0 = tf.cast(Y0[:,1], tf.float64)


            # Diffgleichung-Fehler + Randbedingungen-Fehler + Anfangsbedingungen-Fehler
            
            loss = math.reduce_mean(math.square(fr)) + math.reduce_mean(math.square(fi)) + \
                math.reduce_mean(math.squared_difference(u0,u0_true)) + math.reduce_mean(math.squared_difference(v0,v0_true)) + \
                math.reduce_mean(math.squared_difference(uubx,uubx_true)) + math.reduce_mean(math.squared_difference(vubx,vubx_true)) + \
                math.reduce_mean(math.squared_difference(ulbx,ulbx_true)) + math.reduce_mean(math.squared_difference(vlbx,vlbx_true)) + \
                math.reduce_mean(math.squared_difference(uuby,uuby_true)) + math.reduce_mean(math.squared_difference(vuby,vuby_true)) + \
                math.reduce_mean(math.squared_difference(ulby,ulby_true)) + math.reduce_mean(math.squared_difference(vlby,vlby_true))
                #math.reduce_mean(math.square(uubx)) + math.reduce_mean(math.square(vubx)) + \
                #math.reduce_mean(math.square(ulbx)) + math.reduce_mean(math.square(vlbx)) + \
                #math.reduce_mean(math.square(uuby)) + math.reduce_mean(math.square(vuby)) + \
                #math.reduce_mean(math.square(ulby)) + math.reduce_mean(math.square(vlby))
                    


            grads = g.gradient(loss, self.model.trainable_weights)
    

        return loss, grads


    
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

    
    def print_time(self):

        el_time = self.traindict["el_time"]

        hours = int(np.floor(el_time/3600))
        el_time -= 3600*hours
        minutes = int(np.floor(el_time/60))
        el_time -= 60*minutes
        seconds = int(np.floor(el_time))

        print("Benötigte Zeit: %sh %smin %ss" % (hours,minutes,seconds))

    
    def new_model(self,ub,lb):

        const = (ub - lb) / (self.ub - self.lb)

        inputs = tf.keras.Input(shape=(d_l[0],))
        x = layers.Lambda(lambda x: x[:,0:1], output_shape=(1,))(inputs)
        y = layers.Lambda(lambda x: x[:,1:2], output_shape=(1,))(inputs)
        t = layers.Lambda(lambda x: x[:,2:3], output_shape=(1,))(inputs)

        x = layers.Lambda(lambda x: (x - (ub+lb)/2) / (ub-lb) *2, output_shape=(1,))(x)
        y = layers.Lambda(lambda x: (x - (ub+lb)/2) / (ub-lb) *2, output_shape=(1,))(y)
        t = layers.Lambda(lambda x: (x - (t_end+t_start)/2) / (t_end-t_start) *2, output_shape=(1,))(t)

        x = layers.Concatenate(axis=1)([x,y,t])


        newlayers = self.model.layers[8:]
        first = newlayers[0].get_weights()[0] * const
        newlayers[0].set_weights([first, newlayers[0].get_weights()[1]])

        for v in newlayers:
            x = v(x)

        self.ub = ub
        self.lb = lb

        self.model = tf.keras.Model(inputs=inputs, outputs=x, name="PINN_mod")
    









if __name__ == "__main__": 

    
    lb = -2
    ub = 2

    t_start = 0
    t_end = np.pi * 2

    boundary_number = 10000
    initial_number = 10000

    samples_number = 70000

    d_l = [3, 100, 100, 100, 100, 2]


    epochs = 7500

    learning_rate = 1e-3
    exponential_decay = 0.9



    Schroed_Net = PINN(d_l,ub,lb,t_start,t_end)


    Schroed_Net.model.summary()

    Schroed_Net.train(learning_rate, exponential_decay, epochs, 
                        initial_number, boundary_number, samples_number)


    Schroed_Net.loss_print(append=False)

    Schroed_Net.print_time()

    Schroed_Net.loss_plot()




