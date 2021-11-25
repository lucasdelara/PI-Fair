import numpy as np
import ot
import pandas as pd
import os
import sklearn as sk
from scipy.optimize import minimize
from scipy.special import expit

# METHOD TO IMPORT DATA

def read_and_treat_adult_data(DirectoryName,SensitiveVarName='Sex'):
    """
    Read and treat the adult census data as in [Besse et al., The American Statistician, 2021]
    -> DirectoryName is the directory in which the files "adult.data.csv" and "adult.test.csv" are located
    ->  SensitiveVarName is the string representing the sensitive variable in "adult.data.csv" and "adult.test.csv"
    -> Return [X_train, X_test, y_train, y_test, S_train, S_test,X_col_names]
    """

    #read and merge original data
    original_data_train = pd.read_csv(
        os.path.join(DirectoryName, "adult.data.csv"),
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    original_data_test = pd.read_csv(
        os.path.join(DirectoryName, "adult.test.csv"),
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    original_data = pd.concat([original_data_test,original_data_train])
    original_data.reset_index(inplace = True, drop = True)
    #print(original_data.tail())

    #data preparation 1/3
    data=original_data.copy()

    data['Child'] = np.where(data['Relationship']=='Own-child', 'ChildYes', 'ChildNo')
    data['OrigEthn'] = np.where(data['OrigEthn']=='White', 'CaucYes', 'CaucNo')
    data=data.drop(columns=['fnlwgt','Relationship','Country','Education'])
    data=data.replace('<=50K.','<=50K')
    data=data.replace('>50K.','>50K')

    #print(original_data.tail())

    #data preparation 2/3
    data_ohe=data.copy()
    data_ohe['Target'] = np.where(data_ohe['Target']=='>50K', 1., 0.)
    #print(' -> In column Target: label >50K gets 1.')
    data_ohe['OrigEthn'] = np.where(data_ohe['OrigEthn']=='CaucYes', 1., 0.)
    #print(' -> In column '+str('OrigEthn')+': label '+str('CaucYes')+' gets 1.')

    data_ohe['Sex'] = np.where(data_ohe['Sex']=='Male', 1., 0.)
    #print(' -> In column '+str('Sex')+': label '+str('Male')+' gets 1.')

    for col in ['Workclass', 'Martial Status', 'Occupation', 'Child']:
        if len(set(list(data_ohe[col])))==2:
            LabelThatGets1=data_ohe[col][0]
            data_ohe[col] = np.where(data_ohe[col]==LabelThatGets1, 1., 0.)
            #print(' -> In column '+str(col)+': label '+str(LabelThatGets1)+' gets 1.')
        else:
            #print(' -> In column '+str(col)+': one-hot encoding conversion with labels '+str(set(list(data_ohe[col]))))
            data_ohe=pd.get_dummies(data_ohe,prefix=[col],columns=[col])

    #print(data_ohe.tail())

    #data preparation 3/3
    #... extract the X and y np.arrays
    y=data_ohe['Target'].values.reshape(-1,1)

    data_ohe_wo_target=data_ohe.drop(columns=['Target'])

    X_col_names=list(data_ohe_wo_target.columns)
    X=data_ohe_wo_target.values

    #... split the learning and test samples
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #... print the np.array shapes
    #print('n_train=',X_train.shape[0])
    #print('n_test=',X_test.shape[0])
    #print('p=',X_test.shape[1])

    #... center-reduce the arrays X_train and X_test to make sure all variables have the same scale
    X_train_NoScaling=X_train.copy()
    X_train=sk.preprocessing.scale(X_train)
    X_test_NoScaling=X_test.copy()
    X_test=sk.preprocessing.scale(X_test)

    S_train=X_train_NoScaling[:,X_col_names.index(SensitiveVarName)].ravel()
    S_test=X_test_NoScaling[:,X_col_names.index(SensitiveVarName)].ravel()

    print("S_train.shape:",S_train.shape)
    print("X_train.shape:",X_train.shape)
    print("y_train.shape:",y_train.shape)
    print("S_test.shape:",S_test.shape)
    print("X_test.shape:",X_test.shape)
    print("y_test.shape:",y_test.shape)

    return [X_train, X_test, y_train, y_test, S_train, S_test,X_col_names]

# METHODS TO GENERATE A COUNTERFACTUAL MODEL

def _generate_index(S):
    """Computes the indexes corresponding to each protected group.

    Input
    ----------
    S : 1darray, shape (n_samples,)
        Array of protected attributes.

    Output
    ----------
    group_indexes : list of 1darrays
                    group_indexes[s] contains the indexes of the individuals in group s.

    """
    n_groups=len(set(S))
    groups_indexes = [[] for _ in range(n_groups)]
    for s in range(n_groups):
        groups_indexes[s] = np.where(S==s)[0]
    return groups_indexes

def learn_cf(X,S):
    """Computes the counterfactual model.

    Inputs
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    S : ndarray, shape (n_samples,)
        Array of protected attributes.

    Outputs
    ----------
    model : dict
            defines a counterfactual model.

    """
    
    model = {}
    group_indexes = _generate_index(S)
    model['group_indexes'] = group_indexes
    n_groups=len(group_indexes)
    model['n_groups'] = n_groups
    ot_plans = [[None]*n_groups]*n_groups # stores the n_groups*n_groups transport plans
    group_sizes = []
    for s in range(n_groups):
        ns = len(group_indexes[s])
        group_sizes.append(ns)
        ws = (1/ns)*np.ones(ns) # probability weights of the empirical distribution of the source group
        for t in range(s): # for t=s the transport matrix is useless; for t>s the transport matrix is obtained by symmetry
            nt = len(group_indexes[t])
            wt = (1/nt)*np.ones(nt) # probability weights of the empirical distribution of the target group
            C = ot.dist(X[group_indexes[s]],X[group_indexes[t]]) # computes the cost matrix between the two groups
            pi_st = ot.emd(ws,wt,C,numItermax=1e8) # computes the transport plan s -> t
            I,J = np.where(pi_st>0) # indexes of the non-zero coefficient of the transport matrix
            pi = pi_st.flatten()
            pi = pi[pi>0] # flattened array of the non-zero coefficients of the transport matrix
            ot_plans[s][t] = [pi,I,J] # sparse representation of the the transport plan
            pi_ts = pi_st.T # the transport plan t -> s is obtained by transposition
            I,J = np.where(pi_ts>0) 
            pi = pi_ts.flatten()
            pi = pi[pi>0]
            ot_plans[t][s] = [pi,I,J]
    model['ot_plans'] = ot_plans

    model['group_sizes'] = group_sizes
    return model


## METHODS TO COMPUTE FAIRNESS METRICS

def _sum_axis1(X,I,J):
    """Computes the sum over the second axis (axis=1) of a sparse ndarray flattened across the first two dimensions.

    Inputs
    ----------
    X : ndarray, shape (n_coeff, n_features)
        Flattened array to sum.
        
    I : ndarray, shape (n_coeff,)
        Array of rows with non zero coefficients.

    J : ndarray, shape (n_coeff,)
        Array of columns with non zero coefficients.
    
    Outputs
    ----------
    sigma : ndarray, shape (n_features,)
    """
    K = list(zip(I, J))
    n = len(np.unique(I))
    if X.ndim==1:
        sigma = np.zeros(n)
    else:
        d = X.shape[1]
        sigma = np.zeros((n,d))
    for k in range(len(K)):
        sigma[K[k][0]] = sigma[K[k][0]] + X[k]
    return sigma

def counterfactual_fairness_rate(y_pred,model,epsilon=0,delta=0.1):
    """Computes the counterfactual fairness rate of a sample of predictions.
    WARNING: Works only for two groups.

    Inputs
    ----------
    y_pred : ndarray, shape (n_samples,)
             Array of predictions.

    flat_Pi : ndarray, shape (n_coeff,)
              Flattened counterfactual model.
        
    I : ndarray, shape (n_coeff,)
        Array of rows with non zero coefficients.

    J : ndarray, shape (n_coeff,)
        Array of columns with non zero coefficients.

    epsilon : float
              Tolerance for the gap in disparate treatment.

    delta : float
            Probability threshold for disparate treatment.
    """
    n_samples = len(y_pred)
    group_sizes = model['group_sizes']
    group_indexes = model['group_indexes']
    cases_in_0 = np.zeros(group_sizes[0])
    cases_in_1 = np.zeros(group_sizes[1])
    probability_vector = np.zeros(n_samples)
    ot_plans = model['ot_plans']
    pi_01,I_01,J_01 = ot_plans[0][1]
    pi_10,I_10,J_10 = ot_plans[1][0]
    y0 = y_pred[group_indexes[0]]
    y1 = y_pred[group_indexes[1]]
    tolerance_matrix_1 = (np.abs(y1[I_10]-y0[J_10]) <= epsilon).astype(float)
    cases_in_1 = _sum_axis1(group_sizes[1]*pi_10*tolerance_matrix_1,I_10,J_10)
    tolerance_matrix_0 = (np.abs(y0[I_01]-y1[J_01]) <= epsilon).astype(float)
    cases_in_0 = _sum_axis1(group_sizes[0]*pi_01*tolerance_matrix_0,I_01,J_01)
    cases = np.concatenate((cases_in_0,cases_in_1))
    cases = (cases >= 1-delta).astype(float)
    return np.sum(cases)/n_samples

def disparate_impact(y,S):
    """Computes the disparate impact of a sample of decisions in a binary classification setting with two protected groups.

    Inputs
    ----------
    y: ndarray, shape (n_samples,)
       Array of decisions (0 or 1).

    S : ndarray, shape (n_samples,)
        Array of protected attributes (0 or 1).
    """
    n=1.*y.shape[0]
        
    pi_1=np.mean(S) #estimated P(S=1)
    pi_0=1-pi_1 #estimated P(S=0)
    p_1=np.mean(S*y)   #estimated P(g(X)=1, S=1)
    p_0=np.mean((1-S)*y) #estimated P(g(X)=1, S=0)
    if p_1*pi_0 > p_0*pi_1 and p_1*pi_0 > 0:
        return (p_0*pi_1)/(p_1*pi_0)
    elif pi_1*p_0 > p_1*pi_0 and pi_1*p_0 > 0:
        return (pi_0*p_1)/(pi_1*p_0)
    else:
        return 1

## LOSS FUNCTIONS

def _log_logistic(X):

    """ Source code: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py """

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    Inputs
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function

    Output
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    """
    
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X) # same dimensions and data types

    idx = X>0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out

def _logistic_loss(w, X, y):
    """Computes the logistic loss and its gradient.

    This function is adapted from scikit-learn source code

    Inputs
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    """
    
    return (-np.sum(y*_log_logistic(np.dot(X,w))+(1-y)*_log_logistic(-np.dot(X,w))),(X.T).dot(expit(np.dot(X,w))-y))

def _counterfactual_loss(w, X, model):
    """Computes the counterfactual loss and its gradient.

    Inputs
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : ndarray, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.
    
    model : dict
            Representation of the counterfactual model.
    """
           
    n_groups = model['n_groups']
    group_sizes = model['group_sizes']
    ot_plans = model['ot_plans']
    group_indexes = model['group_indexes']
    value_by_pair = [] # a pair corresponds to a source group s and a target group t
    grad_by_pair = []
    for s in range(n_groups):
        Xs = X[group_indexes[s]]
        ns = group_sizes[s]
        for t in range(n_groups):
            Xt = X[group_indexes[t]]
            nt = group_sizes[t]
            if s!=t:
                pi,I,J = ot_plans[s][t]
                pi=ns*pi # each pair loss is weighted by the size of the source group
                XX = Xs[I]-Xt[J]
                yy = np.dot(XX,w)
                value_by_pair.append(np.sum(pi*np.power(yy,2))) # pair-wise loss
                grad_by_pair.append(2*np.dot(XX.T,pi*yy)) # pair-wise gradient
    
    return (np.sum(value_by_pair),np.sum(grad_by_pair,axis=0))


## METHOD TO TRAIN A COUNTERFACTUALLY FAIR CLASSIFIER

def train_model(x,y,reg=0,model=None):

    """

    Function that trains the model subject to the Pi-counterfactual fairness penalization.
    If no constraints are given, then simply trains an unaltered classifier.

    ----

    Inputs:

    x : ndarray, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    reg : float
          Weight of the regularization.

    ----

    Outputs:

    w: ndarray, shape (n_features+1,)
       Learnt weight vector for the predictor

    """
    X = np.concatenate((np.ones((x.shape[0],1)),x),axis=1) # add an intercept
    
    if reg == 0: # train a standard predictor
    
        def _fun(w,X,y):
            return _logistic_loss(w,X,y)

        w = minimize(fun = _fun,
            x0 = 2*np.random.rand(X.shape[1],)-1,
            args = (X,y),
            method = 'L-BFGS-B',
            jac=True,
            options = {"maxiter":1e7, 'disp':True},
            )
    
    else:
        
            
        def _fun(w,X,y,model):
            ll,lg = _logistic_loss(w,X,y)
            cfl, cfg = _counterfactual_loss(w,X,model)
            return (ll+reg*cfl,lg+reg*cfg)
         
        w = minimize(fun = _fun,
            x0 = 2*np.random.rand(X.shape[1],)-1,
            args = (X,y,model),
            method = 'L-BFGS-B',
            jac=True,
            options = {"maxiter":1e7, 'disp':True}
            )

    return w.x

## METHOD TO MAKE BINARY PREDICTIONS

def classifier(x,w):
    X = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    p_hat = expit(np.dot(X,w))
    return np.round(p_hat)