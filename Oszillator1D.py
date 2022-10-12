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
from tensorflow import keras, math
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import time









class PINN:
    
    def __init__(self,d_l,ub,lb,t_start,t_end, n=[2, 3, 1, 0.5, 0.2], coherent=False, coh_fac=1.4):
        
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
            1: lambda x: tf.cast(2*x, dtype=tf.float64),
            2: lambda x: tf.cast(4*x**2 - 2, dtype=tf.float64),
            3: lambda x: tf.cast(8*x**3 - 12*x, dtype=tf.float64),
            4: lambda x: tf.cast(16*x**4 - 48*x**2 + 12, dtype=tf.float64),
            5: lambda x: tf.cast(32*x**5 - 160*x**3 + 120*x, dtype=tf.float64),
            6: lambda x: tf.cast(64*x**6 - 480*x**4 + 720*x**2 - 120, dtype=tf.float64),
            #7: lambda x: tf.cast(128*x**7 - 1344*x**5 + 3360*x**3 - 1680*x, dtype=tf.float64),
            #8: lambda x: tf.cast(256*x**8 - 3584*x**6 +13440*x**4 - 13440*x**2 + 1680, dtype=tf.float64)
        }

        self.n = n/np.linalg.norm(n)

        self.coherent = coherent
        self.coh_fac = coh_fac

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

        self.calliter = range(150000).__iter__()

        # Erstellen vom Model
        inputs = keras.Input(shape=(d_l[0],))

        x = layers.Lambda(lambda x: x[:,0:1], output_shape=(1,))(inputs)
        t = layers.Lambda(lambda x: x[:,1:2], output_shape=(1,))(inputs)

        x = layers.Lambda(lambda x: (x - (ub+lb)/2) / (ub-lb) *2, output_shape=(1,))(x)
        t = layers.Lambda(lambda x: (x - (t_end+t_start)/2) / (t_end-t_start) *2, output_shape=(1,))(t)

        x = layers.Concatenate(axis=1)([x,t])

        for k in d_l[1:-1]:
            x = layers.Dense(k, activation="tanh")(x)

        outputs = layers.Dense(d_l[-1])(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="PINN")


        


    def train(self, l_r, exp, epochs, i_n, b_n, s_n):

        start_time = time.time()

        opt = keras.optimizers.Adam(learning_rate = keras.optimizers.schedules.ExponentialDecay(l_r,1000,exp))

        # Erstellen der x und t Werte für Training
        x_train = np.random.rand(s_n) * (self.ub-self.lb) + self.lb
        t_train = np.random.rand(s_n) * (self.t_end-self.t_start) + self.t_start

        X_train = np.stack((x_train, t_train), axis=1)
        X = tf.Variable(X_train)

        x0 = np.random.rand(i_n) * (self.ub-self.lb) + self.lb          # x0 ... x Werte bei t=0

        u0_true, v0_true = self.solution(x0, tf.zeros(tf.shape(x0),dtype=tf.float64))

        tb = np.random.rand(b_n) * (self.t_end - self.t_start) + self.t_start
        xb = tf.constant(np.ones(b_n))

        dlb = xb*self.lb            # lb - lower bound, ub - upper bound, Randbedingungen
        dub = xb*self.ub

        ulb_true, vlb_true = self.solution(dlb,tb)
        uub_true, vub_true = self.solution(dub,tb)

        self.traindict.update({
            "X": X,
            "x0": x0,
            "tb": tb,
            "dlb": dlb,
            "dub": dub,
            "epochs": epochs,
            "u0_true": u0_true,
            "v0_true": v0_true,
            "ulb_true": ulb_true,
            "vlb_true": vlb_true,
            "uub_true": uub_true,
            "vub_true": vub_true
        })

        for epoch in range(epochs):


            loss, grads = self.loss_step(X,x0,dlb,dub,tb,u0_true,v0_true,ulb_true,vlb_true,uub_true,vub_true)
                
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



        sopt.minimize(self.soptfunc,weights0,method='L-BFGS-B',jac=True,callback=self.calling,options={'maxiter': 100000,
                                                                                            'maxfun': 100000,
                                                                                            'maxcor': 50,
                                                                                            'maxls': 50,
                                                                                            'ftol' : 1 * np.finfo(float).eps})

        
        self.traindict.update({"el_time": time.time() - start_time})




    
    def roundloss(self,x):
        return round(x, -np.int(np.floor(np.log10(abs(x))))+3)

   

    def solution(self,x,t):  # Lösung, die für Anfangsbedingungen, Randbedingungen & Kontrolle verwendet wird

        if self.coherent:

            cons = np.exp(-1 * np.abs(self.coh_fac)**2)

            herfun = ((self.m*self.omega/np.pi/self.h)**(0.25) * math.exp(-self.m*self.omega*x**2/2/self.h)) * cons
            solfunr = math.cos(t * self.omega/2) * herfun
            solfuni = - math.sin(t * self.omega/2) * herfun

            for k in range(1,len(self.hermites)+1):
                
                herfun = self.coh_fac**k / np.sqrt(np.math.factorial(k)) * cons *\
                    (1/np.sqrt(2**k * np.math.factorial(k)) * (self.m*self.omega/np.pi/self.h)**(0.25) * math.exp(-self.m*self.omega*x**2/2/self.h) * self.hermites[k](np.sqrt(self.m*self.omega/self.h) * x))
                
                solfunr += math.cos(t * self.omega* (k + 0.5)) * herfun
                solfuni -= math.sin(t * self.omega* (k + 0.5)) * herfun

            return tf.cast(solfunr,tf.float64), tf.cast(solfuni,tf.float64)

        else:

            herfun = self.n[0]*((self.m*self.omega/np.pi/self.h)**(0.25) * math.exp(-self.m*self.omega*x**2/2/self.h))
            solfunr = math.cos(t * self.omega/2) * herfun
            solfuni = - math.sin(t * self.omega/2) * herfun

            for k in range(1,len(self.n)):

                herfun = self.n[k] * (1/np.sqrt(2**k * np.math.factorial(k)) * (self.m*self.omega/np.pi/self.h)**(0.25) * math.exp(-self.m*self.omega*x**2/2/self.h) * self.hermites[k](np.sqrt(self.m*self.omega/self.h) * x))
                solfunr += math.cos(t * self.omega* (k + 0.5)) * herfun
                solfuni -= math.sin(t * self.omega* (k + 0.5)) * herfun


            return tf.cast(solfunr,tf.float64), tf.cast(solfuni,tf.float64)


    # Printet Realtei-, Imaginärteil- & Absolutbetragloss
    # Nimmt aber keinen Einfluss auf den Trainingsprozess
    def loss_print(self,append=True):
        ypred = self.model(self.xtest) 

        upred = ypred[:,0].numpy()
        vpred = ypred[:,1].numpy()
        hpred = np.sqrt(upred**2+vpred**2)

        real_loss = np.linalg.norm(upred-self.utrue,2)/np.linalg.norm(self.utrue,2)
        imag_loss = np.linalg.norm(vpred-self.vtrue,2)/np.linalg.norm(self.vtrue,2)
        abs_loss = np.linalg.norm(hpred-self.htrue,2)/np.linalg.norm(self.htrue,2)


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

            loss, grads = self.loss_step(self.traindict["X"],self.traindict["x0"],self.traindict["dlb"],self.traindict["dub"],self.traindict["tb"],
                                            self.traindict["u0_true"],self.traindict["v0_true"],self.traindict["ulb_true"],self.traindict["vlb_true"],
                                            self.traindict["uub_true"],self.traindict["vub_true"])

            print("L-BFGS-B Epoche %s --- Loss: %s" % (count, self.roundloss(float(loss))))

            self.losslist.append(loss)
            self.loss_print()


    
    def soptfunc2(self,weightlist):

        weights = self.list_to_weights(weightlist)             # Nimmt einen 1D-Array und bringt ihn die Form der Weights
        self.model.set_weights(weights)                        # Neue Gewichte werden verwendet

        loss, grads = self.loss_step(self.traindict["X"],self.traindict["x0"],self.traindict["dlb"],self.traindict["dub"],self.traindict["tb"],
                                            self.traindict["u0_true"],self.traindict["v0_true"],self.traindict["ulb_true"],self.traindict["vlb_true"],
                                            self.traindict["uub_true"],self.traindict["vub_true"])

        return self.weights_to_arrays(loss, grads)             # Gradient hat die Form der Gewichte, muss wieder in die Form eines 1D-Tensors gebracht werden



    def soptfunc(self,weightlist):

        return [v.numpy().astype(np.float64) for v in self.soptfunc2(weightlist)]               # Konvertierung zu NumPy-Arrays




    @tf.function
    def loss_step(self,X,x0,xlb,xub,tb,u0_true,v0_true,ulb_true,vlb_true,uub_true,vub_true):

        with tf.GradientTape(persistent=True) as g:

            x = X[:,0]
            t = X[:,1]
                
            g.watch([x,t])

            X2 = tf.stack([x, t], axis=1)
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
            fr = -self.h**2 /2/self.m * u_xx + self.m *self.omega**2 *x**2 *u/2 + self.h * v_t 
            fi = -self.h**2 /2/self.m * v_xx + self.m *self.omega**2 *x**2 *v/2 - self.h * u_t 
                

            
            # Randbedingungen
            Xlb2 = tf.stack([xlb, tb], axis=1)
            Xub2 = tf.stack([xub, tb], axis=1)

            Ylb = self.model(Xlb2)
            Yub = self.model(Xub2)

            ulb = tf.cast(Ylb[:,0], tf.float64)
            vlb = tf.cast(Ylb[:,1], tf.float64)
            uub = tf.cast(Yub[:,0], tf.float64)
            vub = tf.cast(Yub[:,1], tf.float64)

            
            # Anfangsbedingungen
            X02 = tf.stack([x0, tf.zeros(tf.shape(x0),dtype=tf.float64)], axis=1)
            Y0 = self.model(X02)

            u0 = tf.cast(Y0[:,0], tf.float64)
            v0 = tf.cast(Y0[:,1], tf.float64)


            # Diffgleichung-Fehler + Randbedingungen-Fehler + Anfangsbedingungen-Fehler
            loss = math.reduce_mean(math.square(fr)) + math.reduce_mean(math.square(fi)) + \
                math.reduce_mean(math.squared_difference(u0,u0_true)) + math.reduce_mean(math.squared_difference(v0,v0_true)) + \
                math.reduce_mean(math.squared_difference(uub,uub_true)) + math.reduce_mean(math.squared_difference(vub,vub_true)) + \
                math.reduce_mean(math.squared_difference(ulb,ulb_true)) + math.reduce_mean(math.squared_difference(vlb,vlb_true))
                #math.reduce_mean(math.square(uub)) + math.reduce_mean(math.square(vub)) + \
                #math.reduce_mean(math.square(ulb)) + math.reduce_mean(math.square(vlb))
                

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
    



    









if __name__ == "__main__": 

    
    lb = -5
    ub = 5

    t_start = 0
    t_end = np.pi*2

    boundary_number = 40000
    initial_number = 20000

    samples_number = 100000

    d_l = [2, 80, 80, 100, 100, 2]


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

    Schroed_Net.model.save("Oszi1DGauss_0.h5")

    


