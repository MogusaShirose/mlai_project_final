from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CustomCNN model
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

# Load the model
model = CustomCNN()

# Move model to GPU if available
model.to(device)

# Load model weights
model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define class names and confidence threshold
class_names = ['Apple', 'Strawberry']
confidence_threshold = 0.8

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Resize the frame to (64, 64)
    img = cv2.resize(frame, (64, 64))
    
    # Convert frame to RGB and to PIL image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    # Apply same transformations as for training
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img = transform_test(img).unsqueeze(0).to(device)
    
    # Forward pass to get prediction
    outputs = model(img)
    confidences = torch.softmax(outputs, dim=1)
    max_confidence, predicted = torch.max(confidences, 1)
    
    # Determine predicted class and confidence
    if max_confidence.item() >= confidence_threshold:
        predicted_class = class_names[predicted.item()]
        confidence_percent = max_confidence.item() * 100
        display_text = f'Prediction: {predicted_class} ({confidence_percent:.2f}%)'
    else:
        predicted_class = 'Unknown'
        display_text = f'Prediction: {predicted_class}'

    # Display the prediction and confidence on the frame
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
