import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Dataset import Dataset
from models import Models
import time

start_time = time.time()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = Dataset(is_validation=True, device=device)

batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_choice = 'resnet50'  
num_classes = 2  
dropout = 0  
model = Models(model_choice, num_out_classes=num_classes, dropout=dropout)

model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        print(outputs.data.shape)
        print(labels.shape)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Get predicted labels
        _, predicted = torch.max(outputs.data, 1)
        print(_.shape)
        print(predicted.shape)
        total += labels.size(0)
        correct += (predicted == labels[:, 0]).sum().item()

# Calculate average loss and accuracy
average_loss = test_loss / len(test_dataloader)
accuracy = correct / total * 100

end_time = time.time()
execution_time = end_time - start_time

print(f"Average Loss: {average_loss:.4f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Time to execute: {execution_time} seconds")
