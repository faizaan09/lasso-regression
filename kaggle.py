from lasso import LassoRegression
import pandas as pd
from scipy.sparse import csc_matrix
import numpy as np
import pickle as pkl

# import pdb; pdb.set_trace()
class Kaggle():

    def __init__(self,classifier):
        self.clf = classifier

    def load_data(self,X_train_file,Y_train_file,X_val_file,Y_val_file):
        with open(X_train_file) as f:
            df = pd.read_csv(f,sep=' ',header=None)
        df = df.values
        df[:,0]-=1
        df[:,1]-=1
        shape = ( np.max(df[:,1]+1), np.max(df[:,0])+1)

        self.x_train = csc_matrix((df[:,2],(df[:,1],df[:,0])),shape= shape)
        
        with open(X_val_file) as f:
            df = pd.read_csv(f,sep=' ',header=None)
        df = df.values
        df[:,0]-=1
        df[:,1]-=1
        shape = ( np.max(df[:,1])+1, np.max(df[:,0]) +1)

        self.x_val = csc_matrix((df[:,2],(df[:,1],df[:,0])),shape= shape)


        with open(Y_train_file) as f:
            self.y_train = pd.read_csv(f,header=None).values.reshape(-1)
        
        with open(Y_val_file) as f:
            self.y_val = pd.read_csv(f,header=None).values.reshape(-1)

    def submission(self,file='testData.txt'):
        with open(file) as f:
            df = pd.read_csv(f,sep=' ',header=None)
        df = df.values
        df[:,0]-=1
        df[:,1]-=1
        shape = ( np.max(df[:,1]+1), np.max(df[:,0])+1)

        self.test_data = csc_matrix((df[:,2],(df[:,1],df[:,0])),shape= shape)
        with open('2.04828842447.pkl') as f:
            data = pkl.load(f)

        self.clf.weights = data[1]
        self.clf.bias  = data[2]
        Y = self.clf.predict(self.test_data)
        # print "yo"




def main():

    kaggle = Kaggle(LassoRegression())
    kaggle.load_data('trainData.txt','trainLabels.txt','valData.txt','valLabels.txt')
    kaggle.clf.bias = 0
    kaggle.clf.weights = np.random.normal(0,1,kaggle.x_train.shape[0])
    best_config = kaggle.clf.kaggle_regularization_path(kaggle.x_train,kaggle.y_train,kaggle.x_val,kaggle.y_val)
    
    # kaggle.submission()
    # kaggle.clf.get_important_features()
    

if __name__ == '__main__':
    main()