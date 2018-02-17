from lasso import *

# import pdb; pdb.set_trace()
def main():
    N = 250
    dims = 80
    k = 10
    variance = 1
    lasso = LassoRegression()
    X,Y,real_weigh = lasso.generate_data(N,dims,k,variance)
    lasso.weights = np.random.normal(0,1,dims)
    lasso.bias = 0
    lasso.lambda_regularization_path(X,Y,real_weigh,10)

    ## In previous run, Lambda = 719.6634 gave precision 1.0 and recall 1.0 
    ## Using the same Lambda for question Question 4.3.2

    X,Y,real_weigh = lasso.generate_data(N,dims,k,vari=10)
    lasso.weights = np.random.normal(0,1,dims)
    lasso.bias = 0
    lasso.dense_efficient_Lasso(X,Y,Lambda = 719.6634,epsilon = 1e-4)
    # print lasso.precision(real_weigh,lasso.weights), lasso.recall(real_weigh,lasso.weights)
    ## precision dropped from 1.0 to 0.833 and recall remained to be 1
    
    # lambda_regularization_path(X,Y,weigh,bias,real_weigh,10)
    ## If we increase the lambda, then an increase in precision is observed. 
    # Since high variance, we need high lambda to generalise better 



if __name__ == '__main__':
    main()