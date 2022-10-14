from pathlib import PureWindowsPath
import numpy as np
import pandas as pd 
import glob
from tqdm import tqdm
from scipy.signal import medfilt, resample
from pathlib import PureWindowsPath

class DataPrep():
    MAIN_PATH = 'SisFall_dataset/' 
    N_TIMESTAMPS = 36000
    SENSOR_LIST = ["XAD", "YAD", "ZAD", "XR", "YR", "ZR", "XM", "YM", "ZM"]
    CHOSEN_SENSOR = ["XAD", "ZAD", "XR" ,"YR", "ZR"]
    DISCARDED_DICT = {
        "none": ["XAD", "ZAD", "XR" , "YR", "ZR"],
        "x-accel": ["ZAD", "XR" , "YR", "ZR"],
        "z-accel": ["XAD", "XR" , "YR", "ZR"],
        "x-gyro": ["XAD", "ZAD" , "YR", "ZR"],
        "y-gyro": ["XAD", "ZAD", "XR" , "ZR"],
        "z-gyro": ["XAD", "ZAD", "XR" , "YR"],
        "all-gyro": ["XAD", "ZAD"]
        }

    dataset_mem = 0
    reduced_mem = 0 

    def __init__(self,sampling_rate = 200, discarded_channel = "none"):
        self.sampling_rate = sampling_rate 
        self.discarded_channel = discarded_channel 
        self.selected_channels = self.DISCARDED_DICT[discarded_channel]

    def __reduce_mem_usage__(self, df): 
        """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
        """
        start_mem = df.memory_usage().sum() / 1024**2
        # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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
        # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        self.dataset_mem += start_mem
        self.reduced_mem += end_mem

        return df 
    
    def __get_data__(self, path):
        """
        read and processing data
        return data (ndarray) shape = (n_features, n_timestamps)
        """
        df = pd.read_csv(path, delimiter=',', header=None)
        df = self.__reduce_mem_usage__(df)

        df.columns = self.SENSOR_LIST
        df['ZM'] = df['ZM'].replace({';': ''}, regex=True)
        data = df[self.selected_channels].values.T # shape = (n_features, n_timestamps)
        si = (data.shape[-1] // 200) * self.sampling_rate
        data = resample(x=data, num=si, axis=1)
        data = np.pad(data, ((0, 0), (0, self.N_TIMESTAMPS - data.shape[-1])), 'constant') # pad zero
        data = medfilt(data, kernel_size=(1,3))

        return data # shape = (n_features, n_timestamps)

    # get list of metadata from each file
    def __get_meta__(self, path):
        # in case of window OS (we need to change \ to /)
        path = str(PureWindowsPath(path).as_posix())
        f = path.split('/')[-1].replace('.txt', '') # D01_SA01_R01
        # print(f)
        activity, subject, record = f.split('_') # [D01, SA01, R01]
        label = activity[0] # A or D
        return [label, activity, subject, record]

    def load_dataset(self):
        path_list = glob.glob(self.MAIN_PATH+'*/*.txt')
        X, y, meta = [], [], []
        print("Load Dataset")
        print("Sampling rate: {}\nChannel list: {}\n".format(
            self.sampling_rate,
            self.selected_channels))

        #tqdm -  show a progress meter 
        print("Reducing memory usage")
        for path in tqdm(path_list):
            data_ = self.__get_data__(path)
            meta_ = self.__get_meta__(path)
            X.append(data_)
            y.append(meta_[0])
            meta.append(meta_)
        
        print('Memory usage of all dataframe is {:.2f} MB'.format(self.dataset_mem))
        print('Memory usage after optimization is: {:.2f} MB'.format(self.reduced_mem))
        print('Decreased by {:.1f}%'.format(100 * (self.dataset_mem - self.reduced_mem) / self.dataset_mem))

        return np.array(X), np.array(y), np.array(meta)
    
if __name__ == "__main__":
    Preprocess = DataPrep(sampling_rate= 120, discarded_channel="x-gyro")
    X, y, meta = Preprocess.load_dataset() 

