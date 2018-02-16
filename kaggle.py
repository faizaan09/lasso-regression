from lasso import LassoRegression
import pandas as pd
from scipy.sparse import csc_matrix
import numpy as np

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



def main():

    kaggle = Kaggle(LassoRegression())
    kaggle.load_data('trainData.txt','trainLabels.txt','valData.txt','valLabels.txt')
    kaggle.clf.bias = 0
    kaggle.clf.weights = np.random.normal(0,1,kaggle.x_train.shape[0])
    kaggle.clf.kaggle_regularization_path(kaggle.x_train,kaggle.y_train,kaggle.x_val,kaggle.y_val)
    print "yo"
    

if __name__ == '__main__':
    main()