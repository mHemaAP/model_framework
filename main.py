from utils.model_training import trainModel
from utils.common import *
from models.resnet import ResNet18
from datasets.cifar10_dataset import cifar10Set

# try:
#     from epoch.utils import set_seed, model_summary
#     from epoch.utils.experiment import Experiment
#     from epoch.models.resnet import ResNet18
#     from epoch.datasets import CIFAR10
# except ModuleNotFoundError:
#     from utils.misc import set_seed, model_summary
#     from utils.experiment import Experiment
#     from models import *
#     from datasets import *

SEED = 11

# Set seed for reproducibility
torch.manual_seed(SEED)

# Check if GPU/CUDA is available
use_cuda, device = get_device()

if use_cuda:
    torch.cuda.manual_seed(SEED)

if use_cuda:
  #batch_size = 512
  batch_size = 64
else:
  batch_size = 64


cifar10_data = cifar10Set(batch_size, shuffle=True)

model = ResNet18()

# Network Summary
def model_summary(model, input_size=None, depth=10):
    return torchinfo.summary(model, input_size=(batch_size, 3, 32, 32),
                             depth=depth,
                             col_names=["input_size", 
                                        "output_size", 
                                        "num_params",
                                        "kernel_size", 
                                        "params_percent"]) 

def create_model_train_instance(model, dataset, epochs, loss_criterion, scheduler, optimizer):
    return trainModel(model, dataset, epochs=epochs, loss_criterion=loss_criterion, scheduler=scheduler, optimizer=optimizer)    


def main(loss_criterion='CE', epochs=20, scheduler='one_cycle', optimizer ='SGD'):
    
    train_model1 = create_model_train_instance(model, 
                                               cifar10_data, 
                                               epochs=epochs, 
                                               loss_criterion=loss_criterion,
                                               scheduler=scheduler,
                                               optimizer = optimizer)
    train_model1.trainModel()
    train_model1.display_model_stats()
    
    train_model1.show_cifar10_incorrect_predictions()
    train_model1.show_cifar10_incorrect_predictions(enable_grad_cams=True)


if __name__ == '__main__':
    main()