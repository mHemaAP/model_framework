import torch
import torch.optim as optim

def get_sgd_optimizer(model, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer

# The one cycle LR scheduler is set to have the maximum learning rate 
# at the 5th epoch while training the model. The scheduler is set to not 
# have the third phase so that there is no Annihilation
def get_one_cycle_LR_scheduler(optimizer, train_loader, best_lr, epochs):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=best_lr, 
                                                    steps_per_epoch=len(train_loader),
                                                    pct_start=5/epochs, div_factor = 100, 
                                                    three_phase=False, epochs=epochs, 
                                                    anneal_strategy='linear', final_div_factor=100,
                                                    verbose=False)
    return scheduler

def get_reduce_on_plateau_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                     patience=1, 
                                     verbose=True, 
                                     factor=0.1)
    return scheduler

def get_adam_optimizer(model):

    optimizer = optim.Adam(model.parameters(), lr=1e-9, weight_decay=1e-2)
    return optimizer