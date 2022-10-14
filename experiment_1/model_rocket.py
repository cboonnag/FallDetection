import numpy as np
import pandas as pd
import glob
from tslearn.preprocessing import TimeSeriesResampler
from scipy import signal
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.signal import medfilt
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from pyts.multivariate.transformation import MultivariateTransformer
from pyts.transformation import ROCKET




from utilities.preparation import DataPrep

class RocketPrep(DataPrep):
    

if __name__ == "__main__":

    f = open("Results Rocket EXP1.txt", 'w')

    X, y, meta = load_dataset() 
    threshold = 0.39145434515803695

    
    """
    X shape =shape = (n_samples, n_features, n_timestamps)
    y = meta[:,0]
    activities = meta[:,1]
    subjects = meta[:,2]
    records = meta[:,3]

    """
    subjects_id = np.unique(meta[:,2])
    SA_id = subjects_id[:23]
    SE_id = subjects_id[23:]

    # X = X[: , : , : 3000] 

    print(X.shape, y.shape, meta.shape)
    print('\n', subjects_id, '\n', SA_id, '\n', SE_id)

    # leave-one-subject-out + StratifiedKFold
    f = open( "Results Rocket EXP1.txt",'a')

    listacc = [ ]
    listse = [ ]
    listsp = [ ]
    listf1 = [ ]
    n_k =  10000 
    
    for test_subj in SA_id: # leave-one-subject-out
        print('\n===================================')
        f = open( "Results Rocket EXP1.txt",'a')

        print('test subject:', test_subj)
        f.write('TEST SUBJECT :{}\n'.format(str(test_subj))) 
        learn_idxs = np.where(meta[:,2] != test_subj)[0] # list of learning index
        test_idxs = np.where(meta[:,2] == test_subj)[0] # list of test index
        X_learn, y_learn, meta_learn = X[learn_idxs], y[learn_idxs], meta[learn_idxs]
        X_test, y_test, meta_test = X[test_idxs], y[test_idxs], meta[test_idxs]
        
        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        activity_learn = meta_learn[:,1]
        print(activity_learn )

        #gridsearchcv outerloop 

        #key :  0 

        #[ f1 , acc ,  se , sp ]
        result_grid = [0 , 0 , 0 , 0]
        fold_no = 0  

        for train_idxs, val_idxs in cv.split(X_learn, activity_learn):

            fold_no += 1 
            f.write( "\n\n FOLD NUMBER {}\n".format(fold_no ))
            print("Fold Number {} ".format(fold_no))
            train_idxs = train_idxs[0 : len(train_idxs)]
            X_train, y_train, meta_train = X_learn[train_idxs], y_learn[train_idxs], meta_learn[train_idxs]
            X_val, y_val, meta_val = X_learn[val_idxs], y_learn[val_idxs], meta_learn[val_idxs]
        
            
            print("hello")
            print(train_idxs)
            print( len( train_idxs) )
            
            print( val_idxs )
            
            rocket = MultivariateTransformer(ROCKET(n_kernels=n_k))
            logistic  = LogisticRegression()

            pipe = Pipeline(steps=[("rocket", rocket), ("logistic", logistic)])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict_proba(X_val)[:, 1]
            print(type(y_pred))
            y_pred = np.asarray(['F' if val >= threshold else 'D' for val in y_pred])

            tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=['F', 'D']).ravel()

            accuracy = (tn + tp) / (tn + fp + fn + tp)
            se = tp / (tp + fn)
            sp = tn / (tn + fp)

            f1 = f1_score(y_val, y_pred, average='weighted')

            f.write(str((tn, fp, fn, tp)))
            f.write("\n accuracy : {} se : {} sp : {} \n f1 : {} \n ".format(accuracy , se , sp , f1 ))

            print("accuracy : {} se : {} sp : {} \n f1 : {} ".format(accuracy , se , sp , f1 ))
            

            result_grid[0] += f1 
            result_grid[1] += accuracy
            result_grid[2] += se 
            result_grid[3] += sp 

        for i in range(4) : 
            result_grid[i] /= 10 
        f.write("f1 , acc , se , sp \n")
        f.write(str(result_grid))

        f.write("\n TESTING \n")
        #f1 
        rocket = MultivariateTransformer(ROCKET(n_kernels=n_k))
        logistic  = LogisticRegression()
        pipe = Pipeline(steps=[("rocket", rocket), ("logistic", logistic)])

        learn_idxs = learn_idxs[0 : len(learn_idxs) ]
        X_learn, y_learn, meta_learn = X[learn_idxs], y[learn_idxs], meta[learn_idxs]

        pipe.fit(X_learn, y_learn)
        y_pred = pipe.predict_proba(X_test)[:, 1]
        print(type(y_pred))
        y_pred = np.asarray(['F' if val >= threshold else 'D' for val in y_pred])

        tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=['F', 'D']).ravel()
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        f.write("f1 : {} \n".format(f1))
        f.write(str((tn , fp , fn , tp)))
        listf1.append(f1)

        #acc , sp , se 
        rocket = MultivariateTransformer(ROCKET(n_kernels=n_k))
        logistic  = LogisticRegression()

        pipe = Pipeline(steps=[("rocket", rocket), ("logistic", logistic)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict_proba(X_test)[:, 1]
        print(type(y_pred))
        y_pred = np.asarray(['F' if val >= threshold else 'D' for val in y_pred])

        tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=['F', 'D']).ravel()

        
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        se = tp / (tp + fn)
        sp = tn / (tn + fp)

        f.write(" \nacc : {} se : {} sp : {}  \n".format(accuracy, se , sp ))
        f.write(str((tn , fp , fn , tp)))

        listacc.append( accuracy )
        listse.append( se )
        listsp.append( sp )

        f.close()


    f = open( "Results Rocket EXP1.txt",'a')

    f.write("OVERALL \n")

    avgacc  = sum(listacc) / 23 
    sdacc =  sqrt( (i - avgacc)**2/23  for i in listacc ) 
    avgse = sum(listse) / 23 
    sdse =  sqrt( ( i - avgse )**2 / 23 for i in listse ) 
    avgsp = sum(listsp) / 23 
    sdsp = sqrt( ( i - avgsp)**2 / 23 for i in listsp ) 
    avgf1 = sum(listf1) / 23 
    sdf1 = sqrt( ( i - avgf1 )**2 / 23 for i in listf1 ) 


    f.write("accuracy : {} sd : {} \n".format(avgacc , sdacc))
    f.write("se : {} sd : {}\n ".format(avgse , sdse))
    f.write("sp : {} sd : {} \n".format(avgsp , sdsp))
    f.write("f1  : {} sd : {} \n".format(avgf1 , sdf1))



    f.close()