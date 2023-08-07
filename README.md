# model_framework
This is a common modular library that has model framework created in pytorch for ERAv1.


## Code Structure


```bash

├───S12
│   ├───datasets
│   │   │   abstract_dataset.py
│   │   │   cifar10_dataset.py
│   │   │   __init__.py
│   │
│   ├───models
│   │   │   resnet.py
│   │   │   __init__.py
│   │
│   ├───utils
│   │       back_propogation.py
│   │       model_training.py
│   │       scheduler.py
│   │       common.py
│   │       __init__.py
│   ├───main.py

```

### 1. datasets Module
#### 1.1 datasets/abstract_dataset.py Module

The datasets module contains **dataSet** class which creates train and test loaders and can visualise examples with labels.
It also performs basic transforms like Normalize and ToTensorV2.

This class has been developed to support as a base class for multiple datasets. Currently the class has been tested for CIFAR10 dataset. And this class can be extended to support MNIST dataset as well.

#### 1.2 datasets/cifar10_dataset.py Module
This module inherits the above abstract dataset module and applies normalizations and image augmentations specific to cifar10 datasets.
Image Normalization is done using the following factors

```python

    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
```

Image Augmentations used as as follows:

```python
import albumentations as A

            A.ColorJitter(brightness=0, contrast=0.1, 
                          saturation=0.2, hue=0.1, p=0.5),
            A.ToGray(p=0.2),
            A.PadIfNeeded(40, 40, p=1),
            A.RandomCrop(32, 32, p=1),
            
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            A.CoarseDropout(max_holes=1, 
                            max_height=16, 
                            max_width=16, p=1, 
                            fill_value=0),
            A.CenterCrop(32, 32, p=1)


```

### 2. models Module
This module contains the original resnet18 network architecture. Following is the model summary as given after executing in a notebook
 

```
=====================================================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Kernel Shape              Param %
=====================================================================================================================================================================
ResNet                                   [64, 3, 32, 32]           [64, 10]                  --                        --                             --
├─Conv2d: 1-1                            [64, 3, 32, 32]           [64, 64, 32, 32]          1,728                     [3, 3]                      0.02%
├─BatchNorm2d: 1-2                       [64, 64, 32, 32]          [64, 64, 32, 32]          128                       --                          0.00%
├─Sequential: 1-3                        [64, 64, 32, 32]          [64, 64, 32, 32]          --                        --                             --
│    └─BasicBlock: 2-1                   [64, 64, 32, 32]          [64, 64, 32, 32]          --                        --                             --
│    │    └─Conv2d: 3-1                  [64, 64, 32, 32]          [64, 64, 32, 32]          36,864                    [3, 3]                      0.33%
│    │    └─BatchNorm2d: 3-2             [64, 64, 32, 32]          [64, 64, 32, 32]          128                       --                          0.00%
│    │    └─Conv2d: 3-3                  [64, 64, 32, 32]          [64, 64, 32, 32]          36,864                    [3, 3]                      0.33%
│    │    └─BatchNorm2d: 3-4             [64, 64, 32, 32]          [64, 64, 32, 32]          128                       --                          0.00%
│    │    └─Sequential: 3-5              [64, 64, 32, 32]          [64, 64, 32, 32]          --                        --                             --
│    └─BasicBlock: 2-2                   [64, 64, 32, 32]          [64, 64, 32, 32]          --                        --                             --
│    │    └─Conv2d: 3-6                  [64, 64, 32, 32]          [64, 64, 32, 32]          36,864                    [3, 3]                      0.33%
│    │    └─BatchNorm2d: 3-7             [64, 64, 32, 32]          [64, 64, 32, 32]          128                       --                          0.00%
│    │    └─Conv2d: 3-8                  [64, 64, 32, 32]          [64, 64, 32, 32]          36,864                    [3, 3]                      0.33%
│    │    └─BatchNorm2d: 3-9             [64, 64, 32, 32]          [64, 64, 32, 32]          128                       --                          0.00%
│    │    └─Sequential: 3-10             [64, 64, 32, 32]          [64, 64, 32, 32]          --                        --                             --
├─Sequential: 1-4                        [64, 64, 32, 32]          [64, 128, 16, 16]         --                        --                             --
│    └─BasicBlock: 2-3                   [64, 64, 32, 32]          [64, 128, 16, 16]         --                        --                             --
│    │    └─Conv2d: 3-11                 [64, 64, 32, 32]          [64, 128, 16, 16]         73,728                    [3, 3]                      0.66%
│    │    └─BatchNorm2d: 3-12            [64, 128, 16, 16]         [64, 128, 16, 16]         256                       --                          0.00%
│    │    └─Conv2d: 3-13                 [64, 128, 16, 16]         [64, 128, 16, 16]         147,456                   [3, 3]                      1.32%
│    │    └─BatchNorm2d: 3-14            [64, 128, 16, 16]         [64, 128, 16, 16]         256                       --                          0.00%
│    │    └─Sequential: 3-15             [64, 64, 32, 32]          [64, 128, 16, 16]         --                        --                             --
│    │    │    └─Conv2d: 4-1             [64, 64, 32, 32]          [64, 128, 16, 16]         8,192                     [1, 1]                      0.07%
│    │    │    └─BatchNorm2d: 4-2        [64, 128, 16, 16]         [64, 128, 16, 16]         256                       --                          0.00%
│    └─BasicBlock: 2-4                   [64, 128, 16, 16]         [64, 128, 16, 16]         --                        --                             --
│    │    └─Conv2d: 3-16                 [64, 128, 16, 16]         [64, 128, 16, 16]         147,456                   [3, 3]                      1.32%
│    │    └─BatchNorm2d: 3-17            [64, 128, 16, 16]         [64, 128, 16, 16]         256                       --                          0.00%
│    │    └─Conv2d: 3-18                 [64, 128, 16, 16]         [64, 128, 16, 16]         147,456                   [3, 3]                      1.32%
│    │    └─BatchNorm2d: 3-19            [64, 128, 16, 16]         [64, 128, 16, 16]         256                       --                          0.00%
│    │    └─Sequential: 3-20             [64, 128, 16, 16]         [64, 128, 16, 16]         --                        --                             --
├─Sequential: 1-5                        [64, 128, 16, 16]         [64, 256, 8, 8]           --                        --                             --
│    └─BasicBlock: 2-5                   [64, 128, 16, 16]         [64, 256, 8, 8]           --                        --                             --
│    │    └─Conv2d: 3-21                 [64, 128, 16, 16]         [64, 256, 8, 8]           294,912                   [3, 3]                      2.64%
│    │    └─BatchNorm2d: 3-22            [64, 256, 8, 8]           [64, 256, 8, 8]           512                       --                          0.00%
│    │    └─Conv2d: 3-23                 [64, 256, 8, 8]           [64, 256, 8, 8]           589,824                   [3, 3]                      5.28%
│    │    └─BatchNorm2d: 3-24            [64, 256, 8, 8]           [64, 256, 8, 8]           512                       --                          0.00%
│    │    └─Sequential: 3-25             [64, 128, 16, 16]         [64, 256, 8, 8]           --                        --                             --
│    │    │    └─Conv2d: 4-3             [64, 128, 16, 16]         [64, 256, 8, 8]           32,768                    [1, 1]                      0.29%
│    │    │    └─BatchNorm2d: 4-4        [64, 256, 8, 8]           [64, 256, 8, 8]           512                       --                          0.00%
│    └─BasicBlock: 2-6                   [64, 256, 8, 8]           [64, 256, 8, 8]           --                        --                             --
│    │    └─Conv2d: 3-26                 [64, 256, 8, 8]           [64, 256, 8, 8]           589,824                   [3, 3]                      5.28%
│    │    └─BatchNorm2d: 3-27            [64, 256, 8, 8]           [64, 256, 8, 8]           512                       --                          0.00%
│    │    └─Conv2d: 3-28                 [64, 256, 8, 8]           [64, 256, 8, 8]           589,824                   [3, 3]                      5.28%
│    │    └─BatchNorm2d: 3-29            [64, 256, 8, 8]           [64, 256, 8, 8]           512                       --                          0.00%
│    │    └─Sequential: 3-30             [64, 256, 8, 8]           [64, 256, 8, 8]           --                        --                             --
├─Sequential: 1-6                        [64, 256, 8, 8]           [64, 512, 4, 4]           --                        --                             --
│    └─BasicBlock: 2-7                   [64, 256, 8, 8]           [64, 512, 4, 4]           --                        --                             --
│    │    └─Conv2d: 3-31                 [64, 256, 8, 8]           [64, 512, 4, 4]           1,179,648                 [3, 3]                     10.56%
│    │    └─BatchNorm2d: 3-32            [64, 512, 4, 4]           [64, 512, 4, 4]           1,024                     --                          0.01%
│    │    └─Conv2d: 3-33                 [64, 512, 4, 4]           [64, 512, 4, 4]           2,359,296                 [3, 3]                     21.11%
│    │    └─BatchNorm2d: 3-34            [64, 512, 4, 4]           [64, 512, 4, 4]           1,024                     --                          0.01%
│    │    └─Sequential: 3-35             [64, 256, 8, 8]           [64, 512, 4, 4]           --                        --                             --
│    │    │    └─Conv2d: 4-5             [64, 256, 8, 8]           [64, 512, 4, 4]           131,072                   [1, 1]                      1.17%
│    │    │    └─BatchNorm2d: 4-6        [64, 512, 4, 4]           [64, 512, 4, 4]           1,024                     --                          0.01%
│    └─BasicBlock: 2-8                   [64, 512, 4, 4]           [64, 512, 4, 4]           --                        --                             --
│    │    └─Conv2d: 3-36                 [64, 512, 4, 4]           [64, 512, 4, 4]           2,359,296                 [3, 3]                     21.11%
│    │    └─BatchNorm2d: 3-37            [64, 512, 4, 4]           [64, 512, 4, 4]           1,024                     --                          0.01%
│    │    └─Conv2d: 3-38                 [64, 512, 4, 4]           [64, 512, 4, 4]           2,359,296                 [3, 3]                     21.11%
│    │    └─BatchNorm2d: 3-39            [64, 512, 4, 4]           [64, 512, 4, 4]           1,024                     --                          0.01%
│    │    └─Sequential: 3-40             [64, 512, 4, 4]           [64, 512, 4, 4]           --                        --                             --
├─Linear: 1-7                            [64, 512]                 [64, 10]                  5,130                     --                          0.05%
=====================================================================================================================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
Total mult-adds (G): 35.55
=====================================================================================================================================================================
Input size (MB): 0.79
Forward/backward pass size (MB): 629.15
Params size (MB): 44.70
Estimated Total Size (MB): 674.63
=====================================================================================================================================================================

```

### 3. utils Module
#### 3.1 utils/back_propogation.py
This module contains the train and test methods which are used in training the resnet18 model.
The train and test methods perform training and testing respectively on given model and dataset.
They also accumulate loss, accuracy statistics which can be plotted using common helper functions.

#### 3.2 utils/model_training.py
This module contains the trainModel class, which carries out training and testing.
This module also does the following:
1. Configure Optimizer and scheduler
2. Find the best learning rate using pytorch LR finder module
3. Performs train-test iterations for a given number of epochs or a given validation target accuracy.
4. Show Misclassified examples
5. Create GradCAM visualisations of misclassified examples

#### 3.3 utils/common.py
This module contains following miscellaneous functions:
1. Set random seed
2. Check for GPU device
3. Helper functions to show the dataset images
4. Helper functions to show the misclassified images
5. Functions to plot the model statistics and the learning rate trend during training

#### 3.4 utils/scheduler.py
This module contains wrapper functions for various optimizers and schedulers 

### 4. main Module
This module does all the configurations for setting the device/seed, the batch size, number of epochs, network summary configurations and makes the model and the framework around it more modular so that the notebook using this modular framework executed using high level wrappers. 
