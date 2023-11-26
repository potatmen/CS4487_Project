import torch
from Dataset import Dataset
from torch.utils.data import DataLoader

class Predictor:
  def __init__(self, model, criterion, optimizer):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Starting test dataset loading...")

    test_dataset = Dataset(is_validation=True, device=self.device)
    batch_size = 32
    self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print("Testing dataset loaded!")

    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    
    print("Starting train dataset loading...")

    train_dataset = Dataset(is_validation=False, device=self.device)
    self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Training dataset loaded!")



  def train(self, num_epochs=10):
    # Parallelize training across multiple GPUs
    self.model = torch.nn.DataParallel(self.model)
    
    # Set the model to run on the device
    self.model.to(self.device)
    
    # Train the model...
    for epoch in range(num_epochs):
        for inputs, labels in self.train_loader:
            # Move input and label tensors to the device
            inputs = inputs.to(self.device)
            labels = labels[:, 0].to(self.device).long()
    
            # Zero out the optimizer
            self.optimizer.zero_grad()
    
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
    
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            print(loss.item(), end=", ")
        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    print(f'Finished Training, Loss: {loss.item():.4f}')

  def test(self):
    
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in self.test_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            #print(outputs.data.shape)
            #print(labels.shape)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()

            # Get predicted labels
            _, predicted = torch.max(outputs.data, 1)
            #print(_.shape)
            #print(predicted.shape)
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()

    # Calculate average loss and accuracy
    average_loss = test_loss / len(self.test_dataloader)
    accuracy = correct / total * 100

    print(f"Average Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
