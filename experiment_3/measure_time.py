import os 
import numpy as np
import pandas as pd
import glob
from tslearn.preprocessing import TimeSeriesResampler
from scipy import signal
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from pyts.multivariate.transformation import WEASELMUSE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.signal import medfilt
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pickle 
from tslearn.utils import to_pyts_dataset 
import random 
import time 

main_path = 'SisFall_dataset/'
samp_rate = 200
n_timestamps = 36000
sensor = ["XAD", "YAD", "ZAD", "XR", "YR", "ZR", "XM", "YM", "ZM"]
chosen = ["XAD", "ZAD", "XR" ,"YR", "ZR"]

#train model
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
    # df = reduce_mem_usage(df)
    
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

current = 0
current_sample = 0 
# def find_peak() : 

if __name__ == "__main__":

    # f = open("Results Experiment 1.txt", 'w')
    X, y, meta = load_dataset() 

    subjects_id = np.unique(meta[:,2])
    SA_id = subjects_id[:23]
    SE_id = subjects_id[23:]

    X = X[ : , : , : 36000]
    
    print(X.shape, y.shape, meta.shape)
    print('\n', subjects_id, '\n', SA_id, '\n', SE_id)

    window_lengths =  [ 0.1 , 0.3 , 0.5 , 0.7 , 0.9] 
    
    co = 0 
    listtime = [ ]

    for test_subj in SA_id : # leave-one-subject-out
        file_name = "MeasureTimeExperiment3.txt" 
        f = open( file_name ,'a')
        f.write( test_subj + '\n' )

        print('\n===================================')
        print('test subject:', test_subj)
        f.write('TEST SUBJECT :{}\n'.format(str(test_subj))) 
        learn_idxs = np.where(meta[:,2] != test_subj)[0] # list of learning index
        test_idxs = np.where(meta[:,2] == test_subj)[0] # list of test index
        X_learn, y_learn, meta_learn = X[learn_idxs], y[learn_idxs], meta[learn_idxs]
        print(X_learn.shape)
        print(y_learn.shape )

        filename = 'Model/RETUNE-WEASELMUSEMODEL {}.sav'.format(test_subj)
        # if not os.path.isdir(filename) : 
        #     muse = WEASELMUSE(word_size = 4, window_sizes=window_lengths , strategy='entropy')
        #     logistic  = LogisticRegression() 
        #     pipe = Pipeline(steps=[("muse", muse ), ("logistic", logistic)])

        #     pipe.fit(X_learn , y_learn)
        #     pickle.dump(pipe, open(filename, 'wb'))

        # muse = WEASELMUSE(word_size = 4  , window_sizes=window_lengths , strategy='entropy')
        # logistic  = LogisticRegression() 
        # pipe = Pipeline(steps=[("muse", muse ), ("logistic", logistic)])
        # pipe.fit(X_learn, y_learn)
        # filename = 'WEASELMUSEModel SA{}.sav'.format(test_subj)
        # pickle.dump(pipe, open(filename, 'wb'))
        X_test, y_test, meta_test = X[test_idxs], y[test_idxs], meta[test_idxs]
        # # #loop through each activity, trial 
        n , m , t = X_test.shape
        model = pickle.load(open(filename, 'rb'))
        f.close() 
        sublatency =  [ ] 
        subfpr =  [ ]
        subfnr = [ ] 
        sample = random.randint( 0, n - 1 )
        f = open( file_name ,'a')
        print(X_test[sample, 0 ,:].shape)
        print(meta_test[sample][0])

        print(meta_test[sample])
        print(type(meta_test[sample ]) )
        sample_name = meta_test[sample][0] +  meta_test[sample][1] + meta_test[sample][2] + meta_test[sample][3]
        print(meta_test[sample][0] + meta_test[sample][1])
        f.write(sample_name + '\n') 
        print( type( meta_test )) 
    
        window_length = 1500
        #25 ms overlap 
        for space in range( window_length , 3001 , 5 ) : 
    
            sample_test = np.array( X_test[ sample , : ,  max(0,space - window_length   + 1  ) : space ]).reshape(1, 5, window_length - 1 )

            # print(sample_test.shape )


            shape = np.shape( sample_test )
            # print(shape)
            padded_array  = np.zeros((1 , 5 , 36000 ))
            
            padded_array[:shape[0] , :shape[1] , :shape[2]] = sample_test
            sample_test = padded_array 

            start = time.time() 

            model.predict( sample_test)

            end = time.time() 

            test_time = end - start 

            listtime.append(test_time)
            print(test_time )


    avgtesttime = sum(listtime) / len(listtime)
    sdtime = sqrt((i - avgtesttime)**2/ len(listtime) for i in listtime) 

    f.write( "OVERALL TIME : \nAVG : {}  SD : {}".format(avgtesttime , sdtime ))
    f.close() 
            