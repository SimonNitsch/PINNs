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
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import scipy.integrate as sint
import time
import TunnelEq
import TunnelFinite









class PINN:
    
    def __init__(self,d_l,ub,lb,t_start,t_end,A=[1],phi=[0],E=[1]):
        
        self.d_l = d_l
        self.ub = ub
        self.lb = lb

        # ub = upper bound, lb = lower bound, d_l = Dim der Dense Layer

        self.t_start = t_start
        self.t_end = t_end

        self.traindict = {}
        
        self.m = 1
        self.h = 1

        self.losslist = []

        self.calliter = range(0,50000).__iter__()

        self.int_num = 500

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
        print("")
        print("")

    
    def potential(self,x):

        b = TunnelEq.tunnel().potential(x)
        b += self.barrier(x,1000,self.ub+0.2)
        b += self.barrier(x,1000,self.lb-0.2)

        return b


    def barrier(self,x,amp,pos):

        return amp * (math.tanh(1000*(x+0.2-pos)) - math.tanh(1000*(x-0.2-pos)))/2
        


    def train(self, l_r, exp, epochs, i_n, b_n, s_n, bo_n):

        start_time = time.time()

        opt = optimizers.Adam(learning_rate = optimizers.schedules.ExponentialDecay(l_r,1000,exp))

        # Erstellen der x und t Werte für Training
        x_train = np.random.rand(s_n) * (self.ub-self.lb) + self.lb
        t_train = np.random.rand(s_n) * (self.t_end-self.t_start) + self.t_start

        X_train = np.stack((x_train, t_train), axis=1)
        X = tf.Variable(X_train)

        x0 = np.random.rand(i_n) * (self.ub-self.lb) + self.lb          # x0 ... x Werte bei t=0

        tb = np.random.rand(b_n) * (self.t_end - self.t_start) + self.t_start
        xb = tf.constant(np.ones(b_n))

        dlb = xb*(self.lb)           # lb - lower bound, ub - upper bound, Randbedingungen
        dub = xb*(self.ub)

        t_int = np.random.rand(500) * (self.t_end-self.t_start) + self.t_start
        x_int = np.linspace(self.lb,self.ub,num=self.int_num+1)[1:]

        x_border_u = tf.Variable(self.ub + np.random.rand(bo_n)*0.2)
        x_border_l = tf.Variable(self.lb - np.random.rand(bo_n)*0.2)
        t_border = tf.Variable(np.random.rand(bo_n)*(self.t_end-self.t_start) + self.t_start)

        
        sigma = (self.ub-self.lb)/50
        xm = self.lb + (self.ub-self.lb)/4

        u0_true = TunnelFinite.gaussian(x0,xm,sigma)
        

        self.traindict.update({
            "X": X,
            "x0": x0,
            "tb": tb,
            "dlb": dlb,
            "dub": dub,
            "epochs": epochs,
            "u0_true": u0_true,
            "x_border_u": x_border_u,
            "x_border_l": x_border_l,
            "t_border": t_border,
            "x_int": x_int,
            "t_int": t_int
        })

        for epoch in range(epochs):


            loss, grads = self.loss_step(X,x0,dlb,dub,tb,u0_true,x_border_u,x_border_l,t_border)
                
            opt.apply_gradients(zip(grads, self.model.trainable_weights))



         

            if epoch % 20 == 0:
                
                print("Epoche %s/%s --- Loss: %s" % (epoch,epochs,self.roundloss(float(loss))))
                self.losslist.append(loss)


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
                                            self.traindict["u0_true"],self.traindict["x_border_u"],self.traindict["x_border_l"],
                                            self.traindict["t_border"])

            print("L-BFGS-B Epoche %s --- Loss: %s" % (count, self.roundloss(float(loss))))

            self.losslist.append(loss)


    
    def soptfunc2(self,weightlist):

        weights = self.list_to_weights(weightlist)             # Nimmt einen 1D-Array und bringt ihn die Form der Weights
        self.model.set_weights(weights)                        # Neue Gewichte werden verwendet

        loss, grads = self.loss_step(self.traindict["X"],self.traindict["x0"],self.traindict["dlb"],self.traindict["dub"],self.traindict["tb"],
                                        self.traindict["u0_true"],self.traindict["x_border_u"],self.traindict["x_border_l"],
                                            self.traindict["t_border"])

        return self.weights_to_arrays(loss, grads)             # Gradient hat die Form der Gewichte, muss wieder in die Form eines 1D-Tensors gebracht werden



    def soptfunc(self,weightlist):

        return [v.numpy().astype(np.float64) for v in self.soptfunc2(weightlist)]               # Konvertierung zu NumPy-Arrays

    
    def int_sum(self,y):

        diff = (self.ub-self.lb)/self.int_num

        return math.reduce_sum(y) *diff




    @tf.function
    def loss_step(self,X,x0,xlb,xub,tb,u0_true,x_border_u,x_border_l,t_border):

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
            fr = -self.h**2 /2/self.m * u_xx + self.potential(x) * u + self.h * v_t 
            fi = -self.h**2 /2/self.m * v_xx + self.potential(x) * v - self.h * u_t 



            g.watch([x_border_u,x_border_l,t_border])

            Y_border_u = self.model(tf.stack([x_border_u,t_border], axis=1))
            Y_border_l = self.model(tf.stack([x_border_l,t_border], axis=1))

            u_border_u = tf.cast(Y_border_u[:,0], tf.float64)
            v_border_u = tf.cast(Y_border_u[:,1], tf.float64)
            u_border_l = tf.cast(Y_border_l[:,0], tf.float64)
            v_border_l = tf.cast(Y_border_l[:,1], tf.float64)

            u_border_u_x = g.gradient(u_border_u, x_border_u)
            u_border_u_t = g.gradient(u_border_u, t_border)
            u_border_u_xx = g.gradient(u_border_u_x, x_border_u)

            v_border_u_x = g.gradient(v_border_u, x_border_u)
            v_border_u_t = g.gradient(v_border_u, t_border)
            v_border_u_xx = g.gradient(v_border_u_x, x_border_u)

            u_border_l_x = g.gradient(u_border_l, x_border_l)
            u_border_l_t = g.gradient(u_border_l, t_border)
            u_border_l_xx = g.gradient(u_border_l_x, x_border_l)

            v_border_l_x = g.gradient(v_border_l, x_border_l)
            v_border_l_t = g.gradient(v_border_l, t_border)
            v_border_l_xx = g.gradient(v_border_l_x, x_border_l)


            fr_u = -self.h**2 /2/self.m * u_border_u_xx + self.potential(x_border_u) * u_border_u + self.h * v_border_u_t 
            fi_u = -self.h**2 /2/self.m * v_border_u_xx + self.potential(x_border_u) * v_border_u - self.h * u_border_u_t 
            fr_l = -self.h**2 /2/self.m * u_border_l_xx + self.potential(x_border_l) * u_border_l + self.h * v_border_l_t 
            fi_l = -self.h**2 /2/self.m * v_border_l_xx + self.potential(x_border_l) * v_border_l - self.h * u_border_l_t 


            #Überprüfung der Normierung (Integration)
            x_int = self.traindict["x_int"]
            t_int = self.traindict["t_int"]

            X_int, T_int = np.meshgrid(x_int,t_int)
            int_output = []

            for k in range(X_int.shape[0]):

                Y_int = self.model(tf.constant(np.vstack((X_int[k,:],T_int[k,:])).T, dtype=tf.float64))
                Y2_int = tf.cast(Y_int[:,0], tf.float64) ** 2 + tf.cast(Y_int[:,1], tf.float64) ** 2

                int_output.append(self.int_sum(Y2_int))

            int_output = tf.stack(int_output)

            
            # Randbedingungen
            g.watch([xlb,xub])

            Xlb2 = tf.stack([xlb, tb], axis=1)
            Xub2 = tf.stack([xub, tb], axis=1)

            Ylb = self.model(Xlb2)
            Yub = self.model(Xub2)

            ulb = tf.cast(Ylb[:,0], tf.float64)
            vlb = tf.cast(Ylb[:,1], tf.float64)
            uub = tf.cast(Yub[:,0], tf.float64)
            vub = tf.cast(Yub[:,1], tf.float64)

            # ulb_x = g.gradient(ulb, xlb)
            # vlb_x = g.gradient(vlb, xlb)
            # uub_x = g.gradient(uub, xub)
            # vub_x = g.gradient(vub, xub)

            
            # Anfangsbedingungen
            X02 = tf.stack([x0, tf.zeros(tf.shape(x0),dtype=tf.float64)], axis=1)
            Y0 = self.model(X02)

            u0 = tf.cast(Y0[:,0], tf.float64)
            v0 = tf.cast(Y0[:,1], tf.float64)


            # Diffgleichung-Fehler + Randbedingungen-Fehler + Anfangsbedingungen-Fehler
            loss = math.reduce_mean(math.square(fr)) + math.reduce_mean(math.square(fi)) + \
                math.reduce_mean(math.squared_difference(u0,u0_true)) + math.reduce_mean(math.square(v0)) + \
                math.reduce_mean(math.square(fr_u/20000)) + math.reduce_mean(math.square(fi_u/20000)) + \
                math.reduce_mean(math.square(fr_l/20000)) + math.reduce_mean(math.square(fi_l/20000)) + \
                math.reduce_mean(math.square(uub)) + math.reduce_mean(math.square(vub)) + \
                math.reduce_mean(math.square(ulb)) + math.reduce_mean(math.square(vlb)) + \
                math.reduce_mean(math.squared_difference(int_output,1))
                    


            grads = g.gradient(loss, self.model.trainable_weights)
    

        return loss, grads



    
    def loss_plot(self):

        epochs = self.traindict["epochs"]

        x1 = np.arange(0,epochs,20)
        x2 = epochs + np.arange(0,self.calliter.__next__(),25)
        x = np.concatenate((x1,x2))


        plt.figure()
        plt.plot(x,np.stack(self.losslist))

        plt.legend(["NN-Loss"])
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
    t_end = 5

    boundary_number = 5000
    initial_number = 20000

    samples_number = 30000
    border_number = 15000

    d_l = [2, 100, 120, 120, 100, 2]


    epochs = 15000

    learning_rate = 1e-3
    exponential_decay = 0.96



    Schroed_Net = PINN(d_l,ub,lb,t_start,t_end)


    Schroed_Net.model.summary()

    Schroed_Net.train(learning_rate, exponential_decay, epochs, 
                        initial_number, boundary_number, samples_number, border_number)


    Schroed_Net.print_time()

    Schroed_Net.loss_plot()

    Schroed_Net.model.save("GaussTunnel2.h5")




