import torch
import torch.nn.functional as F

train_losses = []
test_losses = []
train_acc = []
test_acc = []

epoch_test_loss = 0

##### Train Function #####
def train(model, device, train_loader, optimizer, loss_criterion, use_l1=False, lambda_l1=5e-4, scheduler=None):

  model.train()

  correct = 0
  processed = 0


  for data, target in train_loader: 
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation 
    # because PyTorch accumulates the gradients on subsequent backward passes. Because of this, 
    # when you start your training loop, ideally you should zero out the gradients so that you 
    # do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = loss_criterion(y_pred, target)
    train_losses.append(loss)

    # L1 regularization
    if use_l1 == True:
        l1 = 0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
        loss = loss + lambda_l1 * l1    

    # Backpropagation
    loss.backward()
    optimizer.step()

    # To apply Learning Rate
    if scheduler:
        scheduler.step()    

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    train_acc.append(100*correct/processed)

  return train_losses, train_acc

### The following 2 functions are added to extract per epoch loss value
### This loss would be used in changing learning rate
def set_epoch_test_loss(test_loss):
    epoch_test_loss = test_loss

def get_epoch_test_loss():
    return epoch_test_loss


##### Test Function #####
def test(model, device, test_loader, loss_criterion):

    model.eval()

    test_loss = 0
    correct = 0


    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_criterion(output, target).item()  # sum up batch loss            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    test_acc.append(100. * correct / len(test_loader.dataset))
    set_epoch_test_loss(test_loss)

    return test_losses, test_acc

### This function is to clear the model train/test statistics after the model is
### trained for the desired number of epochs
def clear_model_stats():
  train_losses.clear()
  test_losses.clear()
  train_acc.clear()
  test_acc.clear()
