from going_modular.configs import configs
from going_modular.utils import compute_iou

import torch


def one_step_train(model, 
                   train_dataloader,
                   loss_fn,
                   optimizer,
                   device):
    
    model = model.to(device)
    model.train()

    train_loss, train_iou = 0, 0
    # for batch, (X, y) in enumerate(train_dataloader):
        
        # X, y = X.to(device), y.to(device)
        # y = torch.where(y == -1, torch.tensor(0), y)

        # #calculate iou
        # # Assuming your target mask is 'target' with shape (batch_size, 1, width, height)
        # logits = model(X)  # Example logits
        # # logits = logits.to(torch.int32)

        # # Assuming your actual target size is (2, 1, 256, 256)
        # target = y.to(torch.int64)  # Example target mask

        # # Convert target to one-hot encoded tensor
        # target_onehot = torch.zeros_like(logits)
        # target_onehot.scatter_(1, target, 1)

        # # Apply softmax to get probabilities along the channel dimension
        # probs = F.softmax(logits, dim=1)

        # # Convert probabilities to one-hot encoded predictions
        # _, predicted = torch.max(probs, 1)

        # # Convert predicted to one-hot encoded tensor
        # predicted_onehot = torch.zeros_like(logits)
        # predicted_onehot.scatter_(1, predicted.unsqueeze(1), 1)

        # # Compute IoU
        # iou = intersection_over_union_multiclass(predicted_onehot, target_onehot)
        # train_iou += iou


        # #calculate loss
        # y = y.reshape(-1).long()

        # y_pred = model(X)
        # # Reshape the target to (batch_size * height * width)
        # y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, configs.NUM_CLASSES)

        # loss = loss_fn(y_pred, y)
        # train_loss += loss.item()
        # # print(f'loss in {batch} is {loss}')
        # # print(f'loss in {batch} is {loss}')

        # optimizer.zero_grad()

        # loss.backward()
        # optimizer.step()

    for i, (inputs, targets) in enumerate(train_dataloader):

        inputs, targets = inputs.to(device), targets.to(device).to(dtype=torch.int64)

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(f'shape outputs:{outputs.shape}')

        # Reshape the target mask to (batch_size, height, width)
        targets = targets.squeeze(1)
        # print(f'shape targets:{targets.shape}')

        # Calculate CrossEntropy loss
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()
        

        loss.backward()
        optimizer.step()

        num_classes = configs['Num_Classes']
        # Convert predictions to class labels
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        # Assuming targets are already numpy arrays
        targets = targets.squeeze(1).cpu().numpy()

        # Calculate mIoU for the current batch
        batch_iou = compute_iou(predictions, targets, num_classes)

        train_iou += batch_iou

    train_loss = train_loss/len(train_dataloader)
    tain_iou = train_iou/len(train_dataloader)

    return train_loss, tain_iou


def one_step_test(model, 
                  test_dataloader,
                  loss_fn,
                  device):
    
    model = model.to(device)
    model.eval()

    test_loss, test_iou = 0, 0
    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(test_dataloader):

            inputs, targets = inputs.to(device), targets.to(device).to(torch.int64)

            outputs = model(inputs)

            # Reshape the target mask to (batch_size, height, width)
            targets = targets.squeeze(1)

            # Calculate CrossEntropy loss
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

            num_classes = configs['Num_Classes']
            # Convert predictions to class labels
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            # Assuming targets are already numpy arrays
            targets = targets.squeeze(1).cpu().numpy()

            # Calculate mIoU for the current batch
            batch_iou = compute_iou(predictions, targets, num_classes)

            test_iou += batch_iou
            
            if i % 2 == 0:
                print(i)

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



