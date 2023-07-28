import os
import torch
from abc import ABC
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
try:
    from ..utils.common import visualize_dataset_images
except ModuleNotFoundError:
    from utils import visualize_dataset_images
#from utils import visualize_dataset_images


class dataSet(ABC):
    mean = None
    std = None
    classes = None
    dataSet_transformation = None
    train_idata_transforms_default = None

    def __init__(self, batch_size=64, train_img_data_transforms=None, normalize=True, shuffle=True):

        self.batch_size = batch_size
        self.img_data_transforms = train_img_data_transforms or self.train_idata_transforms_default
        self.shuffle = shuffle
        self.normalize = normalize
        self.loader_kwargs = {'batch_size': batch_size, 
                              'num_workers': os.cpu_count(), 
                              'pin_memory': True}
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()
        self.data_iter = iter(self.train_loader)

    def get_train_transforms(self):
        if (self.normalize == True):
            train_transforms_list = [A.Normalize(self.mean, self.std)]
        else:
            train_transforms_list = []
        if self.img_data_transforms is not None:
            train_transforms_list += self.img_data_transforms
        train_transforms_list.append(ToTensorV2())
        return A.Compose(train_transforms_list)

    def get_test_transforms(self):
        if (self.normalize == True):
            test_transforms_list = [A.Normalize(self.mean, self.std), ToTensorV2()]
        else:
            test_transforms_list = [ToTensorV2()]
        return A.Compose(test_transforms_list)

    ### Load the train data as to be used by the model
    def get_train_loader(self):

        train_data = self.dataSet_transformation('./data', 
                                             train=True, 
                                             download=True, 
                                             data_alb_transform=self.train_transforms)
        # train dataloader    
        if self.classes is None:
            self.classes = {i: c for i, c in enumerate(train_data.classes)}
        self.train_loader = torch.utils.data.DataLoader(train_data, 
                                                        shuffle=self.shuffle, 
                                                        **self.loader_kwargs)
        return self.train_loader
    
    ### Load the train and test data as to be used by the model 
    def get_test_loader(self):

        test_data = self.dataSet_transformation('./data', 
                                            train=False, 
                                            download=True,
                                            data_alb_transform=self.test_transforms)

        # test dataloader    
        self.test_loader = torch.utils.data.DataLoader(test_data, 
                                                       shuffle=False, 
                                                       **self.loader_kwargs)
        return self.test_loader

    def de_normalise_image(self, img):

        denorm_img = torch.tensor(img, requires_grad=False)

        for dimg, m, s in zip(denorm_img, self.mean, self.std):
            dimg.mul_(s).add_(m)
        return denorm_img    

    def de_transform_image(self, img):

        if (self.normalize == True):        
            img = self.de_normalise_image(img)

        if len(self.mean) == 3:
            return img.permute(1, 2, 0)
        else:
            return img.squeeze(0)


    ### Display the CIFAR10 data images
    def show_dataset_images(self, figsize=None):

        batch_data, batch_label = next(self.data_iter)
        display_image_list = list()
        display_label_list = list()

        for i in range(len(batch_data)):
            tmp_img = self.de_transform_image(batch_data[i])
            tmp_label = batch_label[i].item()

            if self.classes is not None:
                tmp_label = f'{tmp_label}-{self.classes[tmp_label]}'

            display_image_list.append(tmp_img)
            display_label_list.append(tmp_label)

        visualize_dataset_images(display_image_list,
                                    display_label_list,
                                    ' Data Set Sample Images ',
                                    figsize=figsize)
         
