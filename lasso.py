from numpy.random import normal
from random import choice
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import norm

class LassoRegression():

    def generate_data(self,N,dims,k,vari):
        X = normal(0,1,(dims,N))
        
        choices = np.random.choice([10,-10],k)
        weigh = np.zeros(dims)
        weigh[:choices.shape[0]] = choices
        init_bias = np.zeros(N)

        epsi = normal(0,vari,(N))
        weigh.reshape(dims,-1)
        y = np.matmul(X.T,weigh) + init_bias + epsi
        return X,y,weigh

    def dense_efficient_Lasso(self,X,Y, Lambda,epsilon):
        c = 0
        a = 2*np.sum(np.dot(X,X.T),1)

        old_bias =  self.bias
        res = Y - np.matmul(X.T,self.weights) - self.bias
        loss = np.sum(res**2) + Lambda*(np.linalg.norm(self.weights,1))
        old_loss = float('inf')

        while old_loss - loss > epsilon:
            res = Y - (np.matmul(X.T,self.weights) + self.bias)
            old_bias, self.bias = self.bias, np.mean(res) + old_bias
            res = res + (old_bias - self.bias)
            old_w = self.weights
            for k in xrange(X.shape[0]):
                c = 2*np.matmul(X[k,:],res + self.weights[k]*X[k,:])

                if c < -1*Lambda:
                    self.weights[k] = (c+ Lambda)/a[k]
                
                elif c > Lambda:

                    self.weights[k] = (c - Lambda)/a[k]
                else:
                    self.weights[k] = 0 
                
                res = res + (old_w[k] - self.weights[k])*X[k,:]
            
            loss, old_loss =  np.sum(res**2) + Lambda*(norm(self.weights,1)), loss
            print old_loss, loss
        if old_loss < loss:
            self.weights = old_w


    def efficient_Lasso(self,X,Y, Lambda,epsilon):
        
        bias  = self.bias
        weights = self.weights.copy()
        c = 0
        a = scp.sum(X.power(2),1)*2
        old_bias =  bias
        res = Y - X.T.dot(weights) - bias

        loss = scp.sum(res**2) + Lambda*(np.linalg.norm(weights,1))

        old_loss = float('inf')
        count = 0
        while old_loss - loss > epsilon:
            count+=1            
            res = Y - (X.T.dot(weights) + bias)
            
            res = res.reshape(-1,1)
            old_bias, bias = bias, res.mean() + old_bias
            
            res = res + (old_bias - bias)
            
            old_w = weights.copy()
            for k in xrange(X.shape[0]):

                c = 2* ( X[k] * (res + (weights[k]*X[k]).T) )[0,0]
            
                if c < -1*Lambda:
                    weights[k] = (c+ Lambda)/a[k]
                
                elif c > Lambda:

                    weights[k] = (c - Lambda)/a[k]
                elif c < Lambda and c > -1*Lambda:
                    weights[k] = 0 
                
                res = res + (old_w[k] - weights[k])*X[k].T
            
            loss, old_loss =  (res.dot(res.T) + Lambda*(norm(weights,1)))[0,0], loss
            print count
        
        if old_loss < loss:
            self.weights = old_w
            self.bias = old_bias
        else:
            self.weights = weights
            self.bias = bias
    
    def get_rmse(self,y_true,y_pred):
        rmse = sum([(i-j)**2 for i,j in zip(y_true,y_pred)])
        rmse/=y_true.shape[0]

        return rmse**0.5

    def kaggle_regularization_path(self,x_train,y_train,x_val,y_val,steps=10):
        Lambda = 2* norm( x_train*(y_train - np.mean(y_train) ),np.inf )
        old_val_rmse = float('inf')
        while rmse <= old_val_rmse:
            old_val_rmse = rmse_val 
            self.efficient_Lasso(x_train,y_train,Lambda,1e-4)
            rmse_train = self.get_rmse(y_train, self.predict(x_train) )
            rmse_val = self.get_rmse(y_val, self.predict(x_val) )
            print "###############################\nEnd of Iteration"
            print Lambda, rmse_train,rmse_val, np.count_nonzero(self.weights)
            Lambda/=2
 


    def lambda_regularization_path(self,X,Y,real_weigh,steps=10):
        Lambda = 2* np.linalg.norm( np.matmul(X,(Y - np.mean(Y) )),np.inf )
        plot_data = {'lambda':[],'prec':[],'recall':[]}
        for i in xrange(steps):
            self.dense_efficient_Lasso(X,Y,Lambda,1e-4)
            print Lambda, self.precision(real_weigh,self.weights), self.recall(real_weigh,self.weights)
            plot_data['lambda'].append(Lambda)
            plot_data['prec'].append(self.precision(real_weigh,self.weights))
            plot_data['recall'].append(self.recall(real_weigh,self.weights))
            Lambda/=2
        self.plot_stats(plot_data)            

    def precision(self,true,pred):
        TP = 0
        P = np.count_nonzero(pred)
        if not P:
            return 1
        for i,j in zip(true,pred):
            TP+= (j != 0 and i != 0)

        return TP*1.0/P

    def recall(self,true,pred):
        TP = 0
        k = np.count_nonzero(true)
        if not k:
            return 1
        for i,j in zip(true,pred):
            TP+= (j != 0 and i != 0)
        return TP*1.0/k


    def predict(self,X):
        return X.T*self.weights + self.bias

    def plot_stats(self,plot_data):
        plt.figure(figsize=(10,7))
        plt.plot(plot_data['lambda'],plot_data['prec'],'r')
        plt.plot(plot_data['lambda'],plot_data['recall'],'b')
        plt.xlabel('Lambda regularization path')
        plt.legend(['Precision','Recall'],loc='best')
        plt.xticks([i for i in xrange(0,int(max(plot_data['lambda'])),800)])
        plt.show()    

