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
import scipy.io
import scipy.optimize as sopt
import time









class PINN:
    
    def __init__(self,d_l,ub,lb,t_start,t_end,xtest,utrue,vtrue,htrue):
        
        self.d_l = d_l
        self.ub = ub
        self.lb = lb

        # ub = upper bound, lb = lower bound, d_l = Dim der Dense Layer

        self.t_start = t_start
        self.t_end = t_end

        self.traindict = {}
        
        # Kontrollwerte
        
        self.xtest = xtest
        self.utrue = utrue
        self.vtrue = vtrue
        self.htrue = htrue
  

        self.losslist = []
        self.reallist = []
        self.imaglist = []
        self.abslist = []

        self.calliter = range(0,50000).__iter__()

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

        tb = np.random.rand(b_n) * (self.t_end - self.t_start) + self.t_start
        xb = tf.Variable(np.ones(b_n))

        dlb = xb*self.lb            # lb - lower bound, ub - upper bound, Randbedingungen
        dub = xb*self.ub

        self.traindict.update({
            "X": X,
            "x0": x0,
            "tb": tb,
            "dlb": dlb,
            "dub": dub,
            "epochs": epochs
        })

        for epoch in range(epochs):


            loss, grads = self.loss_step(X,x0,dlb,dub,tb)
                
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

    def initial_functiontf(self,x):
        ifun = 2/math.cosh(x)
        return tf.cast(ifun,tf.float64)


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

            loss, grads = self.loss_step(self.traindict["X"],self.traindict["x0"],self.traindict["dlb"],self.traindict["dub"],self.traindict["tb"])

            print("L-BFGS-B Epoche %s --- Loss: %s" % (count, self.roundloss(float(loss))))

            self.losslist.append(loss)
            self.loss_print()


    
    def soptfunc2(self,weightlist):

        weights = self.list_to_weights(weightlist)             # Nimmt einen 1D-Array und bringt ihn die Form der Weights
        self.model.set_weights(weights)                        # Neue Gewichte werden verwendet

        loss, grads = self.loss_step(self.traindict["X"],self.traindict["x0"],self.traindict["dlb"],self.traindict["dub"],self.traindict["tb"])

        return self.weights_to_arrays(loss, grads)             # Gradient hat die Form der Gewichte, muss wieder in die Form eines 1D-Tensors gebracht werden



    def soptfunc(self,weightlist):

        return [v.numpy().astype(np.float64) for v in self.soptfunc2(weightlist)]               # Konvertierung zu NumPy-Arrays




    @tf.function
    def loss_step(self,X,x0,xlb,xub,tb):

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
            fr = -v_t + 0.5 * u_xx + u * (math.square(u) + math.square(v))
            fi = u_t + 0.5 * v_xx + v * (math.square(u) + math.square(v))
                

            
            g.watch([xub,xlb])
            
            # Randbedingungen ub & lb und deren x Ableitungen müssen übereinstimmen
            Xlb2 = tf.stack([xlb, tb], axis=1)
            Xub2 = tf.stack([xub, tb], axis=1)

            Ylb = self.model(Xlb2)
            Yub = self.model(Xub2)

            ulb = tf.cast(Ylb[:,0], tf.float64)
            vlb = tf.cast(Ylb[:,1], tf.float64)
            uub = tf.cast(Yub[:,0], tf.float64)
            vub = tf.cast(Yub[:,1], tf.float64)


            ulb_x = g.gradient(ulb, xlb)
            uub_x = g.gradient(uub, xub)

            vlb_x = g.gradient(vlb, xlb)
            vub_x = g.gradient(vub, xub)
            
            # Anfangsbedingungen
            X02 = tf.stack([x0, tf.zeros(tf.shape(x0),dtype=tf.dtypes.float64)], axis=1)
            Y0 = self.model(X02)

            u0 = tf.cast(Y0[:,0], tf.float64)
            v0 = tf.cast(Y0[:,1], tf.float64)



            loss = math.reduce_mean(math.square(fr)) + math.reduce_mean(math.square(fi)) + \
                math.reduce_mean(math.squared_difference(uub,ulb)) + math.reduce_mean(math.squared_difference(vub,vlb)) + \
                math.reduce_mean(math.squared_difference(uub_x,ulb_x)) + math.reduce_mean(math.squared_difference(vub_x,vlb_x)) + \
                math.reduce_mean(math.squared_difference(u0,self.initial_functiontf(x0))) + math.reduce_mean(math.square(v0))
                   
                # initial_functiontf ... Anfangsbedingungen    


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
    t_end = np.pi/2

    boundary_number = 5000
    initial_number = 5000

    samples_number = 25000

    d_l = [2, 80, 80, 80, 80, 2]


    epochs = 5000

    learning_rate = 1e-3
    exponential_decay = 0.9

    A = scipy.io.loadmat("NLS.mat")

    Apsi = A["uu"]
    At = A["tt"].flatten()[:,None]
    Ax = A["x"].flatten()[:,None]

    AX, AT = np.meshgrid(Ax, At)

    Au = np.real(Apsi)
    Av = np.imag(Apsi)

    xtest = np.stack((AX.flatten(), AT.flatten()), axis=1)

    utrue = Au.T.flatten()
    vtrue = Av.T.flatten()
    htrue = np.sqrt(utrue**2+vtrue**2)


    Schroed_Net = PINN(d_l,ub,lb,t_start,t_end,xtest,utrue,vtrue,htrue)


    Schroed_Net.model.summary()

    Schroed_Net.train(learning_rate, exponential_decay, epochs, 
                        initial_number, boundary_number, samples_number)


    Schroed_Net.loss_print(append=False)

    Schroed_Net.print_time()

    Schroed_Net.loss_plot()

    Schroed_Net.model.save("NonLin.h5")




