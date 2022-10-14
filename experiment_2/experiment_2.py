from utilities.preparation import DataPrep
from utilities.model import WeaselMuseModel

class WeaselMuseModel_2(WeaselMuseModel):
    WINDOW_SIZE_GRID = [0.7] # Opt param 
    WORD_SIZE_GRID = [4] # opt param

    
if __name__ == "__main__":
    Preprocess = DataPrep()
    X, y, meta = Preprocess.load_dataset() 
    Model = WeaselMuseModel(X, y, meta)
    discarded_choice = list( Model.DISCARDED_DICT.keys() ) 
    sampling_choice = Model.SAMPLING_CHOICE

    for discarded_channel_ in discarded_choice:
        for sampling_rate_ in sampling_choice:
            Model.config(sampling_rate= sampling_rate_, discarded_channel= discarded_channel_)
            Model.grid_search()


    
    Model.config(sampling_rate= 200, discarded_channel="none")
    Model.grid_search()