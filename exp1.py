from utilities.model import RocketModel

class RocketExp1(RocketModel):
    EXPERIMENT_NUMBER = "1"

def exp1_rocket():
    Model = RocketExp1(sampling_rate = 200, discarded_channel = "none")

    # # set hyperparameter for grid search, or [n_kernel] if dont want to search
    # Model.param_grid(n_kernel_grid = list( range(1000,10001,1000) ))
    # Model.train()

if __name__ == "__main__":
    exp1_rocket()
