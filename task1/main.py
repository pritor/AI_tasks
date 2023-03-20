import os
import torch
import torchvision
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from skimage import io, transform




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

batch_size = 32
writer = SummaryWriter()
trainset = torchvision.datasets.ImageFolder(root='./simpsons_dataset', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def find_classes(self, directory: str):
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x_xxx.ext
            ├── class_x_xxy.ext
            │── class_x_xxz.ext
            └── class_y_123.ext
            ├── class_y_nsdf3.ext
            └── ...
            └── class_y_asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """

        classes = ['_'.join(d.split('_')[0:-1]) for d in os.listdir(directory)]
        classes = np.unique(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # Then the class name and the index value to the corresponding dictionary, such as {'cat': 0, 'dog': 1}
        return classes, class_to_idx  # then return class name and class index

    def make_dataset(self, dir, class_to_idx, extensions,is_valid_file):
        """
                    Return, such as [(image path, category index value corresponding to), (), ...]
        """
        images = []

        for img in os.listdir(dir):
            item = (img, class_to_idx['_'.join(img.split('_')[0:-1])])
            images.append(item)
        return images

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        path = self.root +'/' + path
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



testset = CustomImageFolder(root='./kaggle_simpson_testset/kaggle_simpson_testset', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

train_classes = trainset.class_to_idx
testset_classes = testset.class_to_idx

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.allow_tf32 = True
# torch.cuda.synchronize()



if __name__ == "__main__":
    PATH = './my_mobile_net.pth'
    net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    net.load_state_dict(torch.load(PATH))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.90)
    net.to(device)
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss, epoch)
    writer.flush()
    print('Finished Training')
    writer.close()
    PATH = './my_mobile_net.pth'
    torch.save(net.state_dict(), PATH)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            predicted_classes = []
            for i in predicted:
                predicted_classes.append(list(train_classes.keys())[list(train_classes.values()).index(i.item())])
            labels_classes = []
            for i in labels:
               labels_classes.append(list(testset_classes.keys())[list(testset_classes.values()).index(i.item())])
            total += labels.size(0)
            correct += (np.array(predicted_classes) == np.array(labels_classes)).sum().item()

    print(f'Accuracy of the network on the 1000 test images: {100 * correct // total} %')
