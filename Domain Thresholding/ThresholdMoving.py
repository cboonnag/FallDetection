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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from numpy import argmax
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold


main_path = 'SisFall_dataset/'
samp_rate = 200
n_timestamps = 36000
sensor = ["XAD", "YAD", "ZAD", "XR", "YR", "ZR", "XM", "YM", "ZM"]
chosen = ["XAD", "ZAD", "XR" ,"YR", "ZR"]


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

    # f = open("Results Experiment 1.txt", 'w')

    X, y, meta = load_dataset() 
    # shape (4505, 5, 36000)
    
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
    print('\n', subjects_id, '\n', SA_id, '\n', SE_id)
    window_lengths =  [  0.1 , 0.3 , 0.5 , 0.7 , 0.9 ] 

    # leave-one-subject-out + StratifiedKFold

    jlistthreshold = [ ]
    listthreshold = [ ]
    kf = KFold(n_splits= 10 ,  random_state= 42, shuffle=True)

    f = open("Threshold.txt" , 'w')
    fold_no = 0 
    for train_index, test_index in kf.split(X):
        fold_no += 1 
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    
        # f = open( "NOPADResultsWEASEL_MUSE EXP1.txt",'a')
        print('\n===================================')
        
        word_size = 4 
        # X_train, y_train, meta_train = X_learn[train_idxs], y_learn[train_idxs], meta_learn[train_idxs]
        
        

        muse = WEASELMUSE(word_size = word_size , window_sizes = window_lengths , strategy = "entropy" )
        logistic  = LogisticRegression()

        pipe = Pipeline(steps=[("muse", muse ), ("logistic", logistic)])

        pipe.fit(X_train, y_train)


        #Youden J's stat & Precision recall curve threshold 
        yhat = pipe.predict_proba( X_test )
        yhat = yhat[ : ,  1 ]


        fpr, tpr, thresholds = roc_curve( y_test , yhat ,pos_label = 'F')
        # get the best threshold
        J = tpr - fpr
        ix = argmax(J)
        jbest_thresh = thresholds[ix]
        print('Best Threshold=%f' % (jbest_thresh))
        jlistthreshold.append(jbest_thresh)

        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='Logistic')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig("Threshplot/Youden{}.png".format(str(fold_no)))
        plt.clf()
        print("Youden : "+ str(jbest_thresh))

        f.write("Youden : \n"+ str(jbest_thresh))

        precision, recall, thresholds = precision_recall_curve( y_test, yhat ,pos_label = 'F')
        fscore = (2 * precision * recall) / (precision + recall)
        ix = argmax(fscore)
        # locate the index of the largest f score
        best_thresh = thresholds[ix]
        listthreshold.append(best_thresh)

        # Threshplot
        no_skill = len(y_test[y_test==1]) / len(y_test)
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Logistic')
        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig("Threshplot/Precall{}.png".format(str(fold_no)))
        plt.clf()
        print( "Precision : " + str(best_thresh) )

        f.write("Precision : \n" + str(best_thresh))


        # tn, fp, fn, tp = confusion_matrix(
        # y_val, y_pred, labels=['F', 'D']).ravel()


    avgthresh = sum(listthreshold) / len( listthreshold )
    avgjthresh = sum(jlistthreshold) / len(jlistthreshold)

    f.write("Precision Recall thresh : {}\n".format(avgthresh ))
    f.write("Youden J's : {}\n".format(avgjthresh))

    f.close()

        