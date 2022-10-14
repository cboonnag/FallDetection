from utilities.model import RocketModel

sampling_list = [20, 60, 100, 140, 200]
discarded_list = ["none","x-accel","z-accel","x-gyro","y-gyro","z-gyro", "all-gyro"]
class RocketExp2(RocketModel):
    EXPERIMENT_NUMBER = "2"

def exp2_rocket():
    for sampling_rate_ in sampling_list : 
        for discarded_channel_ in discarded_list:
            Model = RocketExp2(sampling_rate = sampling_rate_, discarded_channel = discarded_channel_)

    # # set hyperparameter for grid search, or [n_kernel] if dont want to search
    # Model.param_grid(n_kernel_grid = list( range(1000,10001,1000) ))
    # Model.train()

if __name__ == "__main__":
    exp2_rocket()