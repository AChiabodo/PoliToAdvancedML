import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import AlexNet_Weights
import torch.nn.functional as F
from torchmetrics import Accuracy
from dataset import PACSDataset
from model import AlexNet
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 7
BATCH_SIZE = 256
LR = 4e-3            # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
NUM_EPOCHS = 30      # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20       # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down
ALPHA = 0.1          # Multiplicative factor for domain classifier loss
LOG_FREQUENCY = 10

def evaluate(model, test_loader, device, num_classes=NUM_CLASSES):
    model.eval()
    mean_accuracy = []
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        class_outputs, _ = model(data)
        mean_accuracy.append(accuracy(class_outputs, label))

    accuracy = torch.mean(torch.stack(mean_accuracy))
    print(f'\nAccuracy on the target domain: {100 * accuracy:.2f}%')

def train(model, train_loader, test_loader, optimizer, scheduler, device, num_epochs, log_frequency, eval_frequency=5):
        #### TRAINING LOOP
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            class_outputs, _ = model(data)
            loss = F.cross_entropy(class_outputs, label)
            loss.backward()
            optimizer.step()
            if batch_idx % log_frequency == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.3f}')
        if epoch % eval_frequency == 0 and epoch != 0:
            evaluate(model=model, test_loader=test_loader, device=device)
    
    evaluate(model=model, test_loader=test_loader, device=device)

def DANNtrain(source_train_loader, target_train_loader,test_loader, model, optimizer, scheduler, device, num_epochs, log_frequency, eval_frequency=5, stabilize_frequency=None):
    if stabilize_frequency is not None:
        stabilize = True
    for epoch in range(num_epochs):
        model.train()
        target_domain_label = torch.ones(BATCH_SIZE).long().to(device)
        source_domain_label = torch.zeros(BATCH_SIZE).long().to(device)
        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_data, source_label = source_data
            target_data, _ = target_data
            source_data = source_data.to(device)
            source_label = source_label.to(device)
            optimizer.zero_grad()

            source_class_outputs, source_domain_outputs = model(source_data)
            loss_class = F.cross_entropy(source_class_outputs, source_label)

            if stabilize and epoch % stabilize_frequency == 0:
                loss = loss_class
                loss.backward()
            else:
                loss_source_domain = F.cross_entropy(source_domain_outputs, source_domain_label)
                
                target_data = target_data.to(device)
                _, target_domain_outputs = model(target_data)
                loss_target_domain = F.cross_entropy(target_domain_outputs, target_domain_label)
                
                loss = loss_class + loss_source_domain + loss_target_domain
                loss.backward()
            
            optimizer.step()
            if batch_idx % log_frequency == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.3f}, lr: {scheduler.get_last_lr()[0]:.6f}')
        scheduler.step()
        if epoch % eval_frequency == 0 and epoch != 0:
            evaluate(model=model, test_loader=test_loader, device=device)
    pass

if __name__ == '__main__':
    #### DATA SETUP
    # Define the transforms to use on images
    dataset_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the Dataset object for training & testing
    train_dataset = PACSDataset(root=os.path.join("labs","lab3","PACS"), split='train', transform=dataset_transform , dataset=PACSDataset.DatasetType.CARTOON)
    test_dataset = PACSDataset(root=os.path.join("labs","lab3","PACS"), split='train', transform=dataset_transform , dataset=PACSDataset.DatasetType.SKETCH)

    # Define the DataLoaders
    source_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    target_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    #### ARCHITECTURE SETUP
    # Create the Network Architecture object
    model = AlexNet(alpha=0.1)
    # Load pre-trained weights
    model.load_state_dict(AlexNet_Weights.IMAGENET1K_V1.get_state_dict(),strict=False)
    # Overwrite the final classifier layer as we only have 7 classes in PACS
    model.change_last_layer(NUM_CLASSES)


    #### TRAINING SETUP
    # Move model to device before passing it to the optimizer
    model = model.to(DEVICE)

    # Create Optimizer & Scheduler objects
    optimizer = torch.optim.SGD(model.parameters(), LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS, eta_min=0)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    DANNtrain(model=model, source_train_loader=source_train_loader, target_train_loader=target_train_loader,test_loader=test_loader, optimizer=optimizer, scheduler=scheduler, device=DEVICE, num_epochs=NUM_EPOCHS, log_frequency=LOG_FREQUENCY,stabilize_frequency=6)

    evaluate(model=model, test_loader=test_loader, device=DEVICE)