from going_modular import configs
from going_modular.utils import intersection_over_union_multiclass
import torch.nn.functional as F
import torch


def one_step_train(model, 
                   train_dataloader,
                   loss_fn,
                   optimizer,
                   device):
    
    model = model.to(device)
    model.train()

    train_loss, train_iou = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        
        X, y = X.to(device), y.to(device)
        y = torch.where(y == -1, torch.tensor(0), y)

        #calculate iou
        # Assuming your target mask is 'target' with shape (batch_size, 1, width, height)
        logits = model(X)  # Example logits
        # logits = logits.to(torch.int32)

        # Assuming your actual target size is (2, 1, 256, 256)
        target = y.to(torch.int64)  # Example target mask

        # Convert target to one-hot encoded tensor
        target_onehot = torch.zeros_like(logits)
        target_onehot.scatter_(1, target, 1)

        # Apply softmax to get probabilities along the channel dimension
        probs = F.softmax(logits, dim=1)

        # Convert probabilities to one-hot encoded predictions
        _, predicted = torch.max(probs, 1)

        # Convert predicted to one-hot encoded tensor
        predicted_onehot = torch.zeros_like(logits)
        predicted_onehot.scatter_(1, predicted.unsqueeze(1), 1)

        # Compute IoU
        iou = intersection_over_union_multiclass(predicted_onehot, target_onehot)
        train_iou += iou


        #calculate loss
        y = y.reshape(-1).long()

        y_pred = model(X)
        # Reshape the target to (batch_size * height * width)
        y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, configs.NUM_CLASSES)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # print(f'loss in {batch} is {loss}')
        # print(f'loss in {batch} is {loss}')

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if batch % 3 == 0 :
             print(
                  f'loss : {loss.item()} | '
                  f'iou : {iou}') 

 

    train_loss = train_loss/len(train_dataloader)
    test_iou = test_iou/len(train_dataloader)

    return train_loss, test_iou


def one_step_test(model, 
                  test_dataloader,
                  loss_fn,
                  device):
    
    model = model.to(device)
    model.eval()

    test_loss, test_iou = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
                        
            X, y = X.to(device), y.to(device)
            y = torch.where(y == -1, torch.tensor(0), y)

            #calculate iou
            # Assuming your target mask is 'target' with shape (batch_size, 1, width, height)
            logits = model(X)  # Example logits
            # logits = logits.to(torch.int32)

            # Assuming your actual target size is (2, 1, 256, 256)
            target = y.to(torch.int64)  # Example target mask

            # Convert target to one-hot encoded tensor
            target_onehot = torch.zeros_like(logits)
            target_onehot.scatter_(1, target, 1)

            # Apply softmax to get probabilities along the channel dimension
            probs = F.softmax(logits, dim=1)

            # Convert probabilities to one-hot encoded predictions
            _, predicted = torch.max(probs, 1)

            # Convert predicted to one-hot encoded tensor
            predicted_onehot = torch.zeros_like(logits)
            predicted_onehot.scatter_(1, predicted.unsqueeze(1), 1)


            y = y.reshape(-1).long()

            y_pred = model(X)
            # Reshape the target to (batch_size * height * width)
            y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, configs.NUM_CLASSES)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Compute IoU
            iou = intersection_over_union_multiclass(predicted_onehot, target_onehot)
            test_iou += iou




    test_iou = test_iou / len(test_dataloader)
    test_loss = test_loss/ len(test_dataloader)

    return test_loss, test_iou



def train(model,
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          device,
          epochs):
    
    results = {
            'train_loss':[],
            'train_iou':[],
            'test_loss':[],
            'test_iou':[]
        }
    
    for epoch in range(epochs):

        train_loss, train_iou = one_step_train(model,
                                                train_dataloader,
                                                loss_fn, 
                                                optimizer,
                                                device)

        test_loss, test_iou = one_step_test(model,
                                            test_dataloader,
                                            loss_fn,
                                            device)

        results['train_loss'].append(train_loss)
        results['train_iou'].append(train_iou)
        results['test_loss'].append(test_loss)
        results['test_iou'].append(test_iou)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_iou: {train_iou:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_iou: {test_iou:.4f}"
        )
        
    return results



