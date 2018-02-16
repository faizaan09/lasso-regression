from scipy.sparse import csc
from numpy.random import normal
from random import choice
import numpy as np
import scipy
import matplotlib.pyplot as plt

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

    def efficient_Lasso(self,X,Y, Lambda,epsilon):
        
        c = 0
        a = 2*np.sum(X*X.T,1)
        
        old_bias =  self.bias
        res = Y - X.T*self.weights - self.bias
        loss = np.sum(res**2) + Lambda*(np.linalg.norm(self.weights,1))
        old_loss = float('inf')

        while old_loss - loss > epsilon:
            res = Y - ((X.T*self.weights) + self.bias)
            old_bias, self.bias = self.bias, np.mean(res) + old_bias
            res = res + (old_bias - self.bias)
            for k in xrange(X.shape[0]):
                c = 2*np.matmul(X[k,:],res + self.weights[k]*X[k,:])
                old_w = self.weights[k]

                if c < -1*Lambda:
                    self.weights[k] = (c+ Lambda)/a[k]
                
                elif c > Lambda:

                    self.weights[k] = (c - Lambda)/a[k]
                else:
                    self.weights[k] = 0 
                
                res = res + (old_w - self.weights[k])*X[k,:]
            
            loss, old_loss =  np.sum(res**2) + Lambda*(np.linalg.norm(self.weights,1)), loss
            
    
    def get_max_lambda(self,X,Y):
        return 2* np.max( X*(Y - np.mean(Y) ) )

    def get_rmse(self,y_true,y_pred):
        rmse = sum([(i-j)**2 for i,j in zip(y_true,y_pred)])
        rmse/=y_true.shape[0]

        return rsme**0.5

    def kaggle_regularization_path(self,x_train,y_train,x_val,y_val,steps=10):
        Lambda = self.get_max_lambda(x_train, y_train)
        while Lambda > 20:
            self.efficient_Lasso(x_train,y_train,Lambda,1e-4)
            rmse_pred = self.get_rmse(y_train, self.predict(x_train) )
            rmse_val = self.get_rmse(y_val, self.predict(x_val) )
            print Lambda, rmse_pred,rmse_val, np.count_nonzero(self.weights)
            
            Lambda/=2
        # self.plot_stats(plot_data)            



    def lambda_regularization_path(self,X,Y,real_weigh,steps=10):
        Lambda = self.get_max_lambda(X, Y)
        plot_data = {'lambda':[],'prec':[],'recall':[]}
        for i in xrange(steps):
            self.efficient_Lasso(X,Y,Lambda,1e-4)
            print Lambda, self.precision(real_weigh,self.weights), self.recall(real_weigh,self.weights)
            plot_data['lambda'].append(Lambda)
            plot_data['prec'].append(self.precision(real_weigh,self.weights))
            plot_data['recall'].append(self.recall(real_weigh,self.weights))
            Lambda/=2
        self.plot_stats(plot_data)            

    def precision(self,true,pred):
        TP = 0
        P = np.count_nonzero(pred)
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


    def predict(X):
        return np.matmul(X.T,self.weights) + self.bias

    def plot_stats(self,plot_data):
        plt.figure(figsize=(10,7))
        plt.plot(plot_data['lambda'],plot_data['prec'],'r')
        plt.plot(plot_data['lambda'],plot_data['recall'],'b')
        plt.xlabel('Lambda regularization path')
        plt.legend(['Precision','Recall'],loc='best')
        plt.xticks([i for i in xrange(0,int(max(plot_data['lambda'])),400)])
        plt.show()    

