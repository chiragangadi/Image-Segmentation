import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Foreground & background mask
img_dir = "Scribbled image.png"  # provide the directory of scribbled image
img_pil = Image.open(img_dir)
img = np.array(img_pil, dtype='float')/255.0  # converting it to array and normalising 
img = img[:,:,0:3]  # adding three channels for RGB

eps = 1e-8
foreground_mask = img.mean(axis=2)>=1.0-eps  # threshold for white scribble
background_mask = img.mean(axis=2)<0+eps     # threshlod for black scribble
plt.imshow(foreground_mask)
plt.figure(figsize=(5,10))
plt.imshow(background_mask)
plt.figure(figsize=(5,10))


# Extracting information (i.e x,y,R,G,B) from scribbled part of image
img_dir="Original image..jpg"
img_pil=Image.open(img_dir)
img= np.array(img_pil, dtype='float')/255.0
img = img[:,:,0:3]
nx,ny,nc = img.shape

def extractInformationFromImage(img, mask):
    indices = np.nonzero(mask)
    N_pixels = len(indices[0])
    pixel_info = np.zeros((5,N_pixels))   # store x,y,R,G,B values of all pixels the user scribbled
    pixel_info[0,:] = indices[0] /nx
    pixel_info[1,:] = indices[1] /ny
    pixel_info[2:5,:] = img[indices[0], indices[1],:].T
    return pixel_info, N_pixels

foreground_pixels, N_fore = extractInformationFromImage(img, foreground_mask) # Foreground pixels info
background_pixels, N_back = extractInformationFromImage(img, background_mask) # Backgraound pixels info
       
labels_fore = np.zeros((1,N_fore))  # Assigning foreground labels as zero
labels_back = np.ones((1,N_back))   # Assigning foreground labels as one

foreground_data = np.concatenate((foreground_pixels, labels_fore), axis=0) 
background_data = np.concatenate((background_pixels, labels_back), axis=0)

image_data = np.concatenate((foreground_data,background_data), axis=1) # combining foreground and background data


# Data transformation for data loader
inputs = image_data.T[:, :-1]
labels = image_data.T[:, -1]
inputs = torch.from_numpy(inputs).float()
labels = torch.from_numpy(labels).float()


# Training Fully connected neural network
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import torch.optim as optim

# Dataloader 
dataset = TensorDataset(Tensor(inputs), Tensor(labels))
train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

# Create an instance of the network
from FCNN import Net
net = Net()

# Define the binary cross-entropy loss
criterion = nn.BCELoss()

# Define the optimizer (e.g. stochastic gradient descent)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training network
for epoch in range(50):
    running_loss = 0.0
    for inputs,labels in train_loader:
        labels = labels.unsqueeze(1)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        #Backpropogation
        loss.backward()
        
        #Optimizing weights
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
                
    # print training statistics 
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, running_loss))
    
    
# Inference of the FCNN network
allPixels, a = extractInformationFromImage(img, np.ones((nx,ny), dtype=bool))  # Extracting info of all pixels
allPixels = torch.from_numpy(allPixels)
allPixels = allPixels.float().T
inferenceResult = net(allPixels) 
inferenceResult = inferenceResult.detach().numpy().reshape((nx,ny))

plt.imshow(inferenceResult)
plt.show()
inferenceResult_W_B = np.where(inferenceResult < 0.5 , 0, 1)  # Thresholded image, black and white
plt.imshow(inferenceResult_W_B, cmap='gray')


####### The result obtained after this network will not give convex object segmented out of image ######
####### To achive the convex segmentation the inference black and white image is passed through convex architecture neural network ######

# Data extraction from inference image
inferenceResult_W_B = np.array(inferenceResult_W_B)
inferenceResult_W_B = np.expand_dims(inferenceResult_W_B, axis=2)

def extractInformationFrom_W_B_Image(img, mask):
    indices = np.nonzero(mask)
    N_pixels = len(indices[0])
    pixel_info = np.zeros((3,N_pixels))   # store x,y,R,G,B values of all pixels in the image
    pixel_info[0,:] = indices[0] /nx
    pixel_info[1,:] = indices[1] /ny
    pixel_info[2,:] = img[indices[0], indices[1],:].T

    return pixel_info, N_pixels

image_data_C, N_fore_C = extractInformationFrom_W_B_Image(inferenceResult_W_B, np.ones((nx,ny), dtype=bool))


# Data transformation for data loader
inputs_C = image_data_C.T[:, :-1]
labels_C = image_data_C.T[:, -1]
inputs_C = torch.from_numpy(inputs_C).float()
labels_C = torch.from_numpy(labels_C).float()
labels_C = labels_1_C.view(-1, 1)


# Training convex neural network architecture

# Dataloader
dataset_C = TensorDataset(Tensor(inputs_C), Tensor(labels_C))
train_loader_C = DataLoader(dataset_C, batch_size=256, shuffle=True)

# Create an instance of the network
from ConvexNet import NetConvex
net_C = NetConvex()

# Define the binary cross-entropy loss
criterion_C = nn.BCELoss()

# Define the optimizer (e.g. stochastic gradient descent)
optimizer_C = optim.Adam(net_C.parameters(), lr=0.001)

# Train the network
for epoch in range(30):
    running_loss_C = 0.0
    for inputs_C,labels_C in train_loader_C:
        
        # Zero the parameter gradients
        optimizer_C.zero_grad()

        # Forward pass
        outputs_C = net_C(inputs_C)
        loss_C = criterion_C(outputs_C, labels_C)
        
        #Backpropogation
        loss_C.backward()
        
        #optimizing weights
        optimizer_C.step()
        
        #constraints on weights to be non negative
        with torch.no_grad():
            net_C.fc2.weight.data.clamp_(0)
            net_C.fc3.weight.data.clamp_(0)

        # Print statistics
        running_loss_C += loss_C.item()
        
        
    # print training statistics 
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, running_loss_C))


# Inference of the FCNN network
allPixels_C, a_C = extractInformationFrom_W_B_Image(inferenceResult_W_B, np.ones((nx,ny), dtype=bool))
allPixels_C = allPixels_C[:-1]
allPixels_C = torch.from_numpy(allPixels_C)
allPixels_C = allPixels_C.float().T
inferenceResult_C = net_C(allPixels_C)  # forward pass inference
inferenceResult_C = inferenceResult_C.detach().numpy().reshape((nx,ny))

plt.imshow(inferenceResult_C)
plt.show()
foreground_C = np.where(inferenceResult_C > 0.5 , 1, 0)  # Thresholding image
plt.imshow(foreground_C, cmap='gray')


#### The image obtained from convex neural network has convex foreground object segmentation ######
