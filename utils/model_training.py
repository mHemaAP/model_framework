import time
import math
import torch
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder

from .back_propogation import train, test
from .scheduler import *
from .common import *

torch.manual_seed(11)


class trainModel():
    """ Class to encapsulate model training """

    def __init__(self, model,
                 dataset, 
                 epochs=20, 
                 loss_criterion='CE',
                 scheduler='one_cycle',
                 optimizer = 'SGD'
                 ):
        
        self.model = model
        _, self.device = get_device()        
        if(optimizer == 'Adam'):
            self.optimizer = get_adam_optimizer(model)
        else:
            self.optimizer = get_sgd_optimizer(model, lr=0.05)

        if(loss_criterion == 'NLL'):  
            self.loss_function = torch.nn.functional.nll_loss()
        elif(loss_criterion == 'CE'):
            self.loss_function = torch.nn.CrossEntropyLoss()        

        self.epochs = epochs
        self.dataset = dataset


        if scheduler == 'one_cycle':
            self.best_lr = self.get_best_lr()
            self.scheduler = get_one_cycle_LR_scheduler(self.optimizer, 
                                                        self.dataset.train_loader, 
                                                        self.best_lr, 
                                                        epochs)
            decay_lr = True

        else:
            self.scheduler = get_reduce_on_plateau_scheduler(self.optimizer)
            decay_lr = False            
            

        self.train = train
            
        self.test = test


        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []

        self.lr_schedule = []

        self.start_time = 0
        self.end_time = 0

        self.best_perc = 90.0
        self.best_path = ""

    def get_best_lr(self):
        find_lr = LRFinder(self.model, self.optimizer, 
                           self.loss_function, device=self.device)
        find_lr.range_test(self.dataset.train_loader, start_lr = 1e-6, end_lr=0.9, 
                             num_iter=200, step_mode='exp')
        ax, best_lr = find_lr.plot()  # to inspect the loss-learning rate graph
        find_lr.reset()  # to reset the model and optimizer to their initial state
        return best_lr        
    
    def epoch_time_period(self):
                
        elapsed_time = self.end_time - self.start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def show_epoch_progress(self, epoch, train_accuracy, train_loss,
                            test_loss, test_accuracy):

        epoch_mins, epoch_secs = self.epoch_time_period()
        learn_rate = self.lr_schedule[epoch]
        print(f'| {epoch+1:5} | {learn_rate:.6f} | {epoch_mins:02}m {epoch_secs:02}s | {train_loss[-1:][0]:.6f}  | {round(train_accuracy[-1:][0], 2):6}%  | {test_loss[-1:][0]:.6f} |{round(test_accuracy[-1:][0], 2):7}% |')
        


    def save_best_model(self, test_accuracy_percent):


        if test_accuracy_percent >= self.best_perc:
            self.best_perc = test_accuracy_percent
            self.best_path = f'model_weights_{test_accuracy_percent:.2f}.pth'
            torch.save(self.model.state_dict(), self.best_path)

    def run_training_model(self):
        
        print(
            f'| Epoch | {"LR":8} | {"Time":7} | {"TrainLoss":7} | {"TrainAcc":7} | {"TestLoss":7} | {"TestAcc":7} |')

        for epoch in range(self.epochs):

            self.start_time = time.time()
            train_loss, train_accuracy = self.train(model=self.model,
                                                    device=self.device,
                                                    train_loader=self.dataset.train_loader, 
                                                    optimizer=self.optimizer, 
                                                    scheduler=self.scheduler,
                                                    loss_criterion = self.loss_function)
            
            test_loss, test_accuracy = self.test(model=self.model,
                                                 device=self.device,
                                                 test_loader=self.dataset.test_loader,
                                                 loss_criterion = self.loss_function)

            
            self.end_time = time.time()
            self.save_best_model(test_accuracy[-1:][0])

            self.lr_schedule.append(self.scheduler.get_last_lr()[0])            

            self.show_epoch_progress(epoch, train_accuracy, train_loss, 
                                      test_loss, test_accuracy)
        
        self.train_loss =  train_loss
        self.test_loss = test_loss
        self.train_accuracy = test_accuracy
        self.test_accuracy = test_accuracy


    def show_best_model(self):

        self.model.load_state_dict(torch.load(self.best_path))
        self.model.eval()

        test_loss, test_accuracy = self.test(self.model, self.device, self.dataset.test_loader, self.loss_function )

        print(f'Test Accuracy: {test_accuracy[-1:][0]:.2f}% | Test Loss: {test_loss[-1:][0]:.6f}')


    ### Display the model statistics         
    def display_model_stats(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        train_loss = [t.cpu().item() for t in self.train_loss]
        axs[0, 0].plot(train_loss)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_accuracy)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_loss)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_accuracy)
        axs[1, 1].set_title("Test Accuracy")

    ### Display the incorrect predictions of the CIFAR10 data images
    def show_cifar10_incorrect_predictions(self, figsize=None, 
                                           denormalize=True, enable_grad_cam=False):
        
        model1_incorrect_pred = get_incorrect_test_predictions(self.model,
                                                    self.device,
                                                    self.dataset.test_loader)
        
        images_list = []
        incorrect_pred_labels_list = []

        for i in range(20):
            img = model1_incorrect_pred["images"][i]
            pred = model1_incorrect_pred["predicted_vals"][i]
            gtruth = model1_incorrect_pred["ground_truths"][i]

            if (enable_grad_cam==True):
                denorm_img = self.dataset.de_transform_image(img).cpu().numpy()

                img = get_gradcam_transform(self.model, img, 
                                      denorm_img, 
                                      pred, self.device)                
            elif(denormalize==True):
                img = self.dataset.de_transform_image(img).cpu()

            if self.dataset.classes is not None:
                pred = f'P{pred}:{self.dataset.classes[pred]}'
                gtruth = f'A{gtruth}:{self.dataset.classes[gtruth]}'
            
            label = f'{pred}::{gtruth}'

            images_list.append(img)
            incorrect_pred_labels_list.append(label)

        visualize_dataset_images(images_list,
                                 incorrect_pred_labels_list,
                                 ' Mis-Classified Images - Predicted Vs Actual ',
                                 figsize=(10, 8)
                                 )
