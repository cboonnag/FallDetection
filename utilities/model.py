# linear algebra and basic package
import numpy as np
from math import sqrt 
from typing import Union, List 

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# time series model
from pyts.multivariate.transformation import MultivariateTransformer
from pyts.multivariate.classification import MultivariateClassifier
from pyts.transformation import ROCKET
from pyts.multivariate.transformation import WEASELMUSE
from pyts.classification import BOSSVS

# evaluation metric 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Data Preparation
from utilities.preparation import DataPrep

# parent class (base class) for model development
class ModelDevelopment():
    # Global
    THRESHOLD = 0.5 
    MODEL_NAME = "MODEL_NAME"
    EXPERIMENT_NUMBER = "0"
    RANDOM_STATE = 42
    param_grid = [0]

    # data logging
    def __logging__(self, log_str):
        self.log_f = open(self.logfile, 'a')
        self.log_f.write(log_str+"\n")
        print(log_str)
        self.log_f.close()

    def __init__(self,sampling_rate = 200, discarded_channel = "none"):
        self.sampling_rate = sampling_rate 
        self.discarded_channel = discarded_channel
        
        #load dataset
        Preprocess = DataPrep(sampling_rate= self.sampling_rate , discarded_channel= self.discarded_channel)
        self.X, self.y, self.meta = Preprocess.load_dataset() 

        self.subjects_id = np.unique(self.meta[:,2])
        self.SA_id = self.subjects_id[:23]
        self.SE_id = self.subjects_id[23:]

        print("\nX shape: {}\ny shape: {}\nmeta shape: {}".format(
            self.X.shape, 
            self.y.shape, 
            self.meta.shape))

        print('\nSubject ID\n', self.subjects_id, '\nSubject ID (adult)\n', self.SA_id, '\nSubject ID (elderly)\n', self.SE_id)

        self.list_acc = [ ]
        self.list_se = [ ]
        self.list_sp = [ ]
        self.list_f1 = [ ]

        self.__set_filename__()
        self.__outer_loop__()

    def __set_filename__(self):
        self.filename = "exp{}_{}_rate={}_discarded={}".format(
            self.EXPERIMENT_NUMBER,
            self.MODEL_NAME, 
            self.sampling_rate,
            self.discarded_channel 
        )
        self.logfile = "logs/exp{}/{}_log.txt".format(self.EXPERIMENT_NUMBER,self.filename)
        self.log_f = open(self.logfile, "w")

    # use this method to create model in child class
    def create_model(self):
        pass

    def __grid_search__(self, 
        train_idxs, val_idxs,
        X_train, y_train, meta_train,
        X_val, y_val, meta_val):
        ''' 
        for param1 in param1_list:
            for param2 in param2_list:
                for param3 in param3_list:
                    self.train(param1, param2, param3)
        '''
        pass


    def __outer_loop__(self):
        
        for test_subj in self.SA_id: #leave-one-subject-out (LOOCV) -- outer loop
            print("\n====================")
            self.log_f = open(self.logfile, "a")
            self.__logging__( "Test subject: {}".format(str(test_subj)) )

            learn_idxs = np.where(self.meta[:,2] != test_subj)[0] # list of learning index
            test_idxs = np.where(self.meta[:,2] == test_subj)[0] # list of test index
            
            X_learn, y_learn, meta_learn = self.X[learn_idxs], self.y[learn_idxs], self.meta[learn_idxs]
            X_test, y_test, meta_test = self.X[test_idxs], self.y[test_idxs], self.meta[test_idxs]

            # training & validation -- inner loop
            optf1, optacc = self.__inner_loop__(X_learn, y_learn, meta_learn) # result of stratified 10 fold cv + gridsearch

            self.__logging__("\n===== TESTING =====\n")
            
            learn_idxs = learn_idxs[0 : len(learn_idxs)]
            X_learn, y_learn, meta_learn = self.X[learn_idxs], self.y[learn_idxs], self.meta[learn_idxs]

            #f1
            pipe = self.create_model(optf1)
            pipe.fit(X_learn, y_learn)
            y_pred = pipe.predict_proba(X_test)[:, 1]
            print(type(y_pred))
            y_pred = np.asarray(['F' if val >= self.THRESHOLD else 'D' for val in y_pred])

            tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=['F', 'D']).ravel()

            f1 = f1_score(y_test, y_pred, average='weighted')
            self.__logging__("f1 : {} \n".format(f1))
            self.__logging__(str((tn , fp , fn , tp)))
            self.list_f1.append(f1) 

            # acc, sp, se
            pipe = self.create_model(optacc)
            pipe.fit(X_learn, y_learn) # X-learn or X-train ******** ??????
            y_pred = pipe.predict_proba(X_test)[:, 1]
            y_pred = np.asarray(['F' if val >= self.THRESHOLD else 'D' for val in y_pred])

            tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=['F', 'D']).ravel()

            accuracy = (tn + tp) / (tn + fp + fn + tp)
            se = tp / (tp + fn)
            sp = tn / (tn + fp)

            self.__logging__(" \nacc : {} se : {} sp : {}  \n".format(accuracy, se , sp ))
            self.__logging__(str((tn , fp , fn , tp)))
            
            self.list_acc.append( accuracy )
            self.list_se.append( se )
            self.list_sp.append( sp )
            
            self.log_f.close()
        
        self.log_f = open(self.logfile,'a')
        self.__logging__("OVERALL\n")

        avgacc  = sum(self.list_acc) / 23 
        sdacc =  sqrt( (i - avgacc)**2/23  for i in self.list_acc ) 
        avgse = sum(self.list_se) / 23 
        sdse =  sqrt( ( i - avgse )**2 / 23 for i in self.list_se ) 
        avgsp = sum(self.list_sp) / 23 
        sdsp = sqrt( ( i - avgsp)**2 / 23 for i in self.list_sp ) 
        avgf1 = sum(self.list_f1) / 23 
        sdf1 = sqrt( ( i - avgf1 )**2 / 23 for i in self.list_f1 ) 


        self.__logging__("accuracy : {} sd : {} \n".format(avgacc , sdacc))
        self.__logging__("se : {} sd : {}\n ".format(avgse , sdse))
        self.__logging__("sp : {} sd : {} \n".format(avgsp , sdsp))
        self.__logging__("f1  : {} sd : {} \n".format(avgf1 , sdf1))
        
        self.log_f.close()
        
    # 10 fold - inner loop
    def __inner_loop__(self, X_learn, y_learn, meta_learn): 
        cv = StratifiedKFold(n_splits=10, random_state= self.RANDOM_STATE, shuffle=True)
        activity_learn = meta_learn[:,1]
        print(activity_learn)

        self.result_grid =  dict( (str(k), np.array([0.0, 0.0, 0.0, 0.0])) for k in self.param_grid)
        fold_no = 0   
        
        # train-validation
        for train_idxs, val_idxs in cv.split(X_learn, activity_learn): # stratified 10 fold -- inner loop
            fold_no += 1 

            #record and print fold no. 
            self.__logging__("Fold Number {}".format(fold_no)) 

            train_idxs = train_idxs[0 : len(train_idxs)]
            X_train, y_train, meta_train = X_learn[train_idxs], y_learn[train_idxs], meta_learn[train_idxs]
            X_val, y_val, meta_val = X_learn[val_idxs], y_learn[val_idxs], meta_learn[val_idxs]

            print("Train-Validataion")
            print("train_idx: ", train_idxs)
            print("len train idx: ", len( train_idxs) )
            print("val idx: ", val_idxs )

            self.__grid_search__(
                train_idxs, val_idxs,
                X_train, y_train, meta_train,
                X_val, y_val, meta_val
                )
        
        #opt param for each 
        maxf1 = -1 
        maxacc = -1 
        #optimal param 
        optf1 = 0 
        optacc = 0  

        for param in self.param_grid : 
            self.result_grid[str(param)] /= 10 # np.array

            #f1
            if  maxf1 < self.result_grid[str(param)][0] : 
                maxf1 = self.result_grid[str(param)][0]
                optf1 = param

            #acc
            if  maxacc < self.result_grid[str(param)][1] : 
                maxacc = self.result_grid[str(param)][1]
                optacc = param

        self.__logging__("\nOPT PARAM Results out of all folds")
        self.__logging__(str(self.result_grid))
        
        self.__logging__("max avgf1: {} Opt param: {}\n".format(maxf1 , optf1 ))
        self.__logging__("max avgacc: {} Opt param: {}\n".format(maxacc , optacc))

        return optf1, optacc
        


    def report():
        pass

class RocketModel(ModelDevelopment):
    #general 
    MODEL_NAME = "ROCKET"
    THRESHOLD = 0.39145434515803695

    #ROCKET
    n_kernels_grid = list( range(1000,10001,1000) )
    n_kernels = 1000 
    param_grid = n_kernels_grid

    def __grid_search__(self, 
        train_idxs, val_idxs,
        X_train, y_train, meta_train,
        X_val, y_val, meta_val):

        for n_kernels in self.n_kernels_grid: 
            self.__logging__("n_kernels: {}".format(n_kernels))
            self.n_kernels = n_kernels

            pipe = self.create_model(self.n_kernels)
            print("fitting ...")
            pipe.fit(X_train, y_train)
            
            print("validating ...")
            y_pred = pipe.predict_proba(X_val)[:, 1]
            y_pred = np.asarray(['F' if val >= self.THRESHOLD else 'D' for val in y_pred])

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=['F', 'D']).ravel()
            
            # evaluation metric 
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            se = tp / (tp + fn)
            sp = tn / (tn + fp)
            f1 = f1_score(y_val, y_pred, average='weighted')

            self.__logging__("f1 score: {}\naccuracy: {}\nsensitivity: {}\nspecificity: {}\n".format(f1, accuracy , se , sp))

            self.result_grid[str(n_kernels)] += np.array([f1, accuracy, se, sp])


    def create_model(self, n_kernels_):
        rocket = MultivariateTransformer(ROCKET(n_kernels= n_kernels_))
        logistic  = LogisticRegression()
        return  Pipeline(steps=[("rocket", rocket), ("logistic", logistic)])