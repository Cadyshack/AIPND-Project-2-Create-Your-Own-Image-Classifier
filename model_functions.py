import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import keep_awake

def model_builder(arch, hidden_units):
    
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters to not backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = None

    if arch == "densenet121" and hidden_units is None:
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 512)),
                                ('relu1', nn.ReLU()),
                                ('drp1', nn.Dropout(0.5)),
                                ('fc2', nn.Linear(512, 256)),
                                ('relu2', nn.ReLU()),
                                ('drp2', nn.Dropout(0.5)),
                                ('fc3', nn.Linear(256, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    elif arch == "densenet121" and hidden_units is not None:
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 512)),
                                ('relu1', nn.ReLU()),
                                ('drp1', nn.Dropout(0.5)),
                                ('fc2', nn.Linear(512, hidden_units)),
                                ('relu2', nn.ReLU()),
                                ('drp2', nn.Dropout(0.5)),
                                ('fc3', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    elif arch == "vgg19" and hidden_units is None:
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 4096)),
                            ('relu1', nn.ReLU()),
                            ('drp1', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(4096, 2043)),
                            ('relu2', nn.ReLU()),
                            ('drp2', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(2043, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    else:
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 4096)),
                            ('relu1', nn.ReLU()),
                            ('drp1', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(4096, hidden_units)),
                            ('relu2', nn.ReLU()),
                            ('drp2', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    model.classifier = classifier
 
    return model

def train_model(model, image_data, device, save_dir, learning_rate, epochs):
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs = epochs
    model.to(device)

    train_losses, validation_losses = [], []
    for e in keep_awake(range(epochs)):
        running_loss = 0
        for images, labels in image_data["trainloader"]:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            validation_loss = 0
            accuracy = 0
            # set model to evaluation mode
            model.eval()
            with torch.no_grad():
                
                for images, labels in image_data["validloader"]:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    validation_loss += criterion(log_ps, labels)
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            train_losses.append(running_loss/len(image_data["validloader"]))
            validation_losses.append(validation_loss/len(image_data["validloader"]))
            
            print(f"Epoch: {e+1}/{epochs}.. ",
                f"Training Loss: {running_loss/len(image_data['trainloader']):.3f}.. ",
                f"Validation Loss: {validation_loss/len(image_data['validloader']):.3f}.. ",
                f"Validation Accuracy: {accuracy/len(image_data['validloader']):.3f}")
            
            # set model back to train mode
            model.train()

def save_checkpoint(arch, model, image_data, save_dir):
    checkpoint = {
                'arch': arch,
                'classifier': model.classifier ,
                'state_dict': model.state_dict(),
                'class_to_idx': image_data['train_data'].class_to_idx
             }
    save_location = save_dir + '/checkpoint2.pth'
    torch.save(checkpoint, save_location)