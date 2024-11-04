import torch
from torchvision import datasets, transforms
import argparse
import torch.nn as nn
import torch.nn.functional as F

# Classifier convoltional neural network
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.001,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()
    
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    
    classifier = Classifier().cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        classifier.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, loss.item()))

        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy: {} %'.format(100 * correct / total))
    
    torch.save(classifier.state_dict(), 'checkpoints/classifier.pth')