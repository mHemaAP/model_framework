import numpy as np
import torch
import torchinfo
import torchvision
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

##### Get Device Details #####
def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)

# def move_loss_to_cpu(loss):
#   moved_loss2cpu = [t.cpu().item() for t in loss]
#   return moved_loss2cpu

#####  Get the count of correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}


#####  Get the incorrect predictions
def GetInCorrectPreds(pPrediction, pLabels):
    pPrediction = pPrediction.argmax(dim=1)
    indices = pPrediction.ne(pLabels).nonzero().reshape(-1).tolist()
    return indices, pPrediction[indices].tolist(), pLabels[indices].tolist()


def get_incorrect_test_predictions(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            ind, pred, truth = GetInCorrectPreds(output, target)
            test_incorrect_pred["images"] += data[ind]
            test_incorrect_pred["ground_truths"] += truth
            test_incorrect_pred["predicted_vals"] += pred

    return test_incorrect_pred

#####  Display the shape and decription of the train data
def display_train_data(train_data):

  print('[Train]')
  print(' - Numpy Shape:', train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))



##### args: points should be of type list of tuples or lists
def plot_learning_rate_trend(curves,title,Figsize = (7,7)):
    fig = plt.figure(figsize=Figsize)
    ax = plt.subplot()
    for curve in curves:
        if("x" not in curve):
            ax.plot(curve["y"], label=curve.get("label", "label"))   
        else:
            ax.plot(curve["x"],curve["y"], label=curve.get("label","label"))
        plt.xlabel(curve.get("xlabel","x-axis"))
        plt.ylabel(curve.get("ylabel","y-axis"))
        plt.title(title)
    ax.legend()
    plt.show()


def visualize_dataset_images(display_image_list, 
                             display_label_list,
                             fig_title, 
                             figsize=None
                             ):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(fig_title, fontsize=18)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        ele_img = display_image_list[i]
        plt.imshow(ele_img, cmap='gray')
        ele_label = display_label_list[i]
        plt.title(str(ele_label))
        plt.xticks([])
        plt.yticks([])

def get_gradcam_transform(model, img_tensor, denorm_img_tensor, pred_label, device):

    grad_cam = GradCAM(model=model, target_layers=[model.layer3[-1]],
                        use_cuda=(device == 'cuda'))

    targets = [ClassifierOutputTarget(pred_label)]

    grayscale_cam = grad_cam(input_tensor=img_tensor.unsqueeze(0), targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    grad_cam_output = show_cam_on_image(denorm_img_tensor, grayscale_cam, use_rgb=True, image_weight=0.7)
    
    return grad_cam_output