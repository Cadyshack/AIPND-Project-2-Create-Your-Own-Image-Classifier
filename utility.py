import torch
from torchvision import datasets, transforms

def load_data(image_dir):
    """
    Retrieve images from path given to image folder which will be used to train the model

    images_dir: 
      relative path to the folder of images that are to be
      classified by the classifier function (string).
      The Image folder is expected to have three sub folders:
        - train
        - valid
        - test
    """
    image_dir = image_dir
    train_dir = image_dir + '/train'
    valid_dir = image_dir + '/valid'
    test_dir = image_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    image_dict = {  "train_data": train_data,
                    "valid_data": valid_data,
                    "test_data": test_data,
                    "trainloader": trainloader,
                    "validloader": validloader,
                    "testloader": testloader
                }
    return image_dict
