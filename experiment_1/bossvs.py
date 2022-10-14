import numpy as np
import pandas as pd
import glob
# from tslearn.preprocessing import TimeSeriesResampler
from scipy import signal
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from pyts.classification import BOSSVS
from pyts.multivariate.classification import MultivariateClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.signal import medfilt
from math import sqrt 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

main_path = 'SisFall_dataset/'
samp_rate = 200
n_timestamps = 36000
sensor = ["XAD", "YAD", "ZAD", "XR", "YR", "ZR", "XM", "YM", "ZM"]
chosen = ["XAD", "ZAD", "XR", "ZR"]


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def get_data(path):
    """
    read and processing data
    return data (ndarray) shape = (n_features, n_timestamps)
    """
    df = pd.read_csv(path, delimiter=',', header=None)
    df = reduce_mem_usage(df)

    df.columns = sensor
    df['ZM'] = df['ZM'].replace({';': ''}, regex=True)
    data = df[chosen].values.T # shape = (n_features, n_timestamps)
    si = (data.shape[-1] // 200) * samp_rate
    data = signal.resample(x=data, num=si, axis=1)
    data = np.pad(data, ((0, 0), (0, n_timestamps-data.shape[-1])), 'constant') # pad zero
    data = medfilt(data, kernel_size=(1,3))
    # data = medfilt(data, kernel_size=1)
    return data # shape = (n_features, n_timestamps)

def get_meta(path):
    """
    get list of metadata from each file
    """
    f = path.split('/')[-1].replace('.txt', '') # D01_SA01_R01
    print(f)
    activity, subject, record = f.split('_') # [D01, SA01, R01]
    label = activity[0] # A or D
    return [label, activity, subject, record]


def load_dataset():
    path_list = glob.glob(main_path+'*/*.txt')
    X, y, meta = [], [], []
    
    for path in tqdm(path_list):
        data_ = get_data(path)
        meta_ = get_meta(path)
        
        X.append(data_)
        y.append(meta_[0])
        meta.append(meta_)
        
    return np.array(X), np.array(y), np.array(meta)

if __name__ == "__main__":

    threshold = 0.39145434515803695
    X, y, meta = load_dataset() 
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

    print(X.shape, y.shape, meta.shape)
    # X = X[: , : , : 6000]
    print('\n', subjects_id, '\n', SA_id, '\n', SE_id)
    
    # leave-one-subject-out + StratifiedKFold

    listacc = [ ]
    listse = [ ]
    listsp = [ ]
    listf1 = [ ]

    for test_subj in SA_id: # leave-one-subject-out
        f = open( "experiment_1/results/bossvs_results_exp1.txt",'a')
        print('\n===================================')
        print('test subject:', test_subj)
        f.write('TEST SUBJECT :{}\n'.format(str(test_subj))) 
        learn_idxs = np.where(meta[:,2] != test_subj)[0] # list of learning index
        test_idxs = np.where(meta[:,2] == test_subj)[0] # list of test index
        # learn_idxs = learn_idxs[0 : len(learn_idxs) - 1000]
        X_learn, y_learn, meta_learn = X[learn_idxs], y[learn_idxs], meta[learn_idxs]
        X_test, y_test, meta_test = X[test_idxs], y[test_idxs], meta[test_idxs] 

        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        activity_learn = meta_learn[:,1]
        print(activity_learn )

        #gridsearchcv outerloop 
        word_sizes = [ 2 , 4 , 6 ]

        #key :  0 

        #[ f1 , acc ,  se , sp ]
        result_grid = {'2' : [0 , 0  , 0 ,0 ] , '4' : [0 , 0,0 ,0 ] , '6' : [0 ,0 ,0,0 ]  }

        fold_no = 0  

        for train_idxs, val_idxs in cv.split(X_learn, activity_learn):

            fold_no += 1 
            f.write( "\n\n FOLD NUMBER {}\n".format(fold_no ))
            print("Fold Number {} ".format(fold_no))
            train_idxs = learn_idxs[0 : len(train_idxs)]
            X_train, y_train, meta_train = X_learn[train_idxs], y_learn[train_idxs], meta_learn[train_idxs]
            X_val, y_val, meta_val = X_learn[val_idxs], y_learn[val_idxs], meta_learn[val_idxs]
        

            for word_size in word_sizes : 

                f.write("\n Word size = {}\n ".format(word_size ))
                print("Word_size  = {} ".format(word_size))

                print("hello")
                print(train_idxs)
                print( len( train_idxs) )
                
                print( val_idxs )
                bossvs = MultivariateClassifier( BOSSVS( word_size =  word_size , window_size = 10  , strategy = "entropy" ))
                pipe = Pipeline(steps=[("bossvs", bossvs)])
 
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)
                print("YES IT WORKS")

                tn, fp, fn, tp = confusion_matrix(
                y_val, y_pred, labels=['F', 'D']).ravel()

                
                accuracy = (tn + tp) / (tn + fp + fn + tp)
                se = tp / (tp + fn)
                sp = tn / (tn + fp)

                f1 = f1_score(y_val, y_pred, average='weighted')

                f.write(str((tn, fp, fn, tp)))
                f.write("\n accuracy : {} se : {} sp : {} \n f1 : {} \n ".format(accuracy , se , sp , f1 ))

                print("accuracy : {} se : {} sp : {} \n f1 : {} ".format(accuracy , se , sp , f1 ))
                

                result_grid[str(word_size)][0] += f1 
                result_grid[str(word_size)][1] += accuracy
                result_grid[str(word_size)][2] += se 
                result_grid[str(word_size)][3] += sp


        
        #opt param for each 
        maxf1 = -1 
        maxacc = -1 
        #optimal param 
        optf1 = 0 
        optacc = 0  

        for i in range(4) : 

            for word_size in word_sizes : 

                result_grid[str(word_size)][i] /= 10 

                if ( i == 0 ) and maxf1  <  result_grid[str(word_size)][i] : 
                    maxf1 = result_grid[str(word_size)][i]
                    optf1 = word_size 
            

                elif ( i == 1 ) and maxacc < result_grid[str(word_size)][i] : 
                    maxacc = result_grid[str(word_size)][i]
                    optacc = word_size


        f.write("\nOPT PARAM Results out of all folds\n".format()) 

        f.write(str(result_grid))
        f.write("max avgf1 : {} Opt word_size : {} \n".format(maxf1 , optf1 ))
        f.write("max avgacc : {} Opt word_size : {}\n ".format(maxacc , optacc))


        f.write("\n TESTING \n")

        learn_idxs = learn_idxs[0 : len(learn_idxs)]
        X_learn, y_learn, meta_learn = X[learn_idxs], y[learn_idxs], meta[learn_idxs]
        bossvs = MultivariateClassifier( BOSSVS(word_size=  word_size , window_size=  8 , strategy = "entropy" ))


        pipe = Pipeline(steps=[("bossvs", bossvs)])
        


        pipe.fit(X_learn, y_learn)
        y_pred = pipe.predict(X_test)

        

        tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=['F', 'D']).ravel()

        f1 = f1_score(y_test, y_pred, average='weighted')
        f.write("f1 : {} \n".format(f1))
        f.write(str((tn , fp , fn , tp)))
        listf1.append(f1)


        #acc , sp , se 
        bossvs = MultivariateClassifier( BOSSVS(word_size=  word_size ,window_size = 8 , strategy = "entropy" ))


        pipe = Pipeline(steps=[("bossvs", bossvs)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

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

    f = open( "bossvs_results_exp1.txt",'a')

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
