import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations with enhanced data augmentation for training
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomRotation(degrees=20),  # Increase the rotation range to 20 degrees
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Custom dataset paths
train_dataset_path = "C:/Users/seanh/Downloads/mlai_FINAL/dataset/train"
test_dataset_path = "C:/Users/seanh/Downloads/mlai_FINAL/dataset/test"

data_train = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform_train)
data_test = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform_test)

# Initialize data loaders with adjusted batch size
train_loader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=False)

# Manually set class weights (example)
class_weights = torch.tensor([0.5, 0.5], device=device)  # Adjust weights based on dataset imbalance

# Define a customized CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*4*4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)  # Output units for the number of classes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model
cnn = CustomCNN()

# Move model to GPU if available
cnn.to(device)

# Define the loss function (cross-entropy loss) with manually set class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define the optimizer (Adam optimizer) with a reduced learning rate
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Initialize metrics function
def compute_metrics(outputs, labels):
    # Convert model outputs to predicted labels
    _, preds = torch.max(outputs, 1)
    
    # Convert labels and predictions to numpy arrays
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    
    # Calculate accuracy, F1-score, precision, and recall
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted', zero_division=1)
    recall = recall_score(labels, preds, average='weighted', zero_division=1)
    
    return accuracy, f1, precision, recall

# Number of epochs to train the model
num_epochs = 25

# Lists to store training and validation metrics
train_accuracies = []
val_accuracies = []

# Early stopping parameters
early_stopping_patience = 7
early_stopping_counter = 0
best_val_accuracy = 0.0

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move inputs and labels to the GPU
        inputs, labels = inputs.to(device), labels.to(device)     
        
        # Initialize parameter gradients
        optimizer.zero_grad()
        
        # Forward pass: compute model output
        outputs = cnn(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Calculate metrics
        accuracy, f1, precision, recall = compute_metrics(outputs, labels)
        running_accuracy += accuracy
        running_f1 += f1
        running_precision += precision
        running_recall += recall
    
    # Calculate average metrics for each epoch
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = running_accuracy / len(train_loader)
    avg_f1 = running_f1 / len(train_loader)
    avg_precision = running_precision / len(train_loader)
    avg_recall = running_recall / len(train_loader)
    
    # Store training accuracy
    train_accuracies.append(avg_accuracy)
    
    # Set model to evaluation mode
    cnn.eval()
    
    # Initialize evaluation metrics
    eval_loss = 0.0
    eval_accuracy = 0.0
    eval_f1 = 0.0
    eval_precision = 0.0
    eval_recall = 0.0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass: compute model output
            outputs = cnn(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Accumulate evaluation loss
            eval_loss += loss.item()
            
            # Calculate metrics
            accuracy, f1, precision, recall = compute_metrics(outputs, labels)
            eval_accuracy += accuracy
            eval_f1 += f1
            eval_precision += precision
            eval_recall += recall
    
    # Calculate average metrics for the evaluation set for each epoch
    avg_eval_loss = eval_loss / len(test_loader)
    avg_eval_accuracy = eval_accuracy / len(test_loader)
    avg_eval_f1 = eval_f1 / len(test_loader)
    avg_eval_precision = eval_precision / len(test_loader)
    avg_eval_recall = eval_recall / len(test_loader)
    
    # Store validation accuracy
    val_accuracies.append(avg_eval_accuracy)
    
    # Print average metrics for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'accuracy: {avg_accuracy:.4f} - f1_score: {avg_f1:.4f} - loss: {avg_loss:.4f} - '
          f'precision: {avg_precision:.4f} - recall: {avg_recall:.4f} - '
          f'val_accuracy: {avg_eval_accuracy:.4f} - val_f1_score: {avg_eval_f1:.4f} - val_loss: {avg_eval_loss:.4f} - '
          f'val_precision: {avg_eval_precision:.4f} - val_recall: {avg_eval_recall:.4f}')
    
    # Learning rate scheduling
    scheduler.step(avg_eval_loss)
    
    # Early stopping
    if avg_eval_accuracy > best_val_accuracy:
        best_val_accuracy = avg_eval_accuracy
        early_stopping_counter = 0
        # Save the best model
        torch.save(cnn.state_dict(), "best_model.pth")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
    
    # Set model back to training mode
    cnn.train()

# Plot training and validation accuracies
plt.figure()
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print('Finished Training')
