#!/home/romi/anaconda3/bin/python3

import torch, torchvision
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time




# In[11]:


class Classifier(nn.Module):
    def __init__(self,n_classes):
        super(Classifier, self).__init__()
        self.resnet =  models.resnet34(pretrained = False)
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(256,n_classes) # 6 is number of classes
        self.relu = nn.LeakyReLU()
    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print('Device: ',device)
#classifier = Classifier(n_classes).to(device)
#summary(classifier,(3,150,150)) #summary is used to create summary of our model similar to keras summary.


def main():

    
    im_size=256
    mean=0.5
    std=0.5
    batch_size=16
    model_dir='/home/romi/'
    #base_dir='/content/drive/My Drive/'
    valid_percentage=0.2
    criterion = nn.CrossEntropyLoss()
    n_classes=6

    init_instant = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print('Device: ',device)
    classifier = Classifier(n_classes).to(device)
    classifier.load_state_dict(torch.load(model_dir+'checkpoint.pt'))
    model_loaded_instant =time.time()

    # In[57]:


    start_image_path='/home/romi/ORB_SLAM/bin/Tracking.jpg'
    end_image_path='/home/romi/ORB_SLAM/bin/TrackLost.png'


    # In[58]:


    start_image=Image.open(start_image_path)
    start_image_gray=start_image.convert(mode='L')

    end_image=Image.open(end_image_path)
    end_image_gray=end_image.convert(mode='L')
    image_loaded_instant = time.time()

    dummy_channel=Image.new(mode='L',size=end_image_gray.size)

    merged_image=Image.merge(mode='RGB',bands=(start_image_gray,end_image_gray,dummy_channel))
    
    test_transforms = transforms.Compose([
        transforms.Resize(size=im_size),
        transforms.CenterCrop(size=im_size), # Image net standards
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std))])


    # In[35]:


    labels={
        #0: 'To recover track , Please move in the forward direction',
        #1: 'To recover track , Please do Clockwise rotation',
        #2: 'To recover track , Please do Anti-Clockwise rotation',
        #3: 'To recover track , Please move in the backward direction',
        #4: 'To recover track , Please move Right',
        #5: 'To recover track , Please move Left'
        0: 'Forward',
        1: 'CW',
        2: 'CCW',
        3: 'Back',
        4: 'Right',
        5: 'Left'
    }


    # In[59]:


    sm = nn.Softmax(dim = 1)

    data=test_transforms(merged_image)
    data.unsqueeze_(0)
    data=Variable(data)
    data = data.type(torch.cuda.FloatTensor)
    image_preprocessed_instant = time.time()

    classifier.eval()
    output = classifier(data)
    image_classified_instant = time.time()
    #print('Here')
    print("\n\nClassifier commands the agent to move : ",labels[output.cpu().data.numpy().argmax()])
    fd = "/home/romi/abc2.txt"
    file = open(fd, 'w') 
    file.write(labels[output.cpu().data.numpy().argmax()]) 
    file.close() 
    file = open(fd, 'r') 
    text = file.read() 
    #print(text) 
    #print('End')
    print('Model Loading Time: ',model_loaded_instant-init_instant,' seconds')
    print('Image Loading Time: ',image_loaded_instant-model_loaded_instant,' seconds')
    print('Image Preprocessing Time: ',image_preprocessed_instant-image_loaded_instant,' seconds')
    print('Image Classification Time: ',image_classified_instant-image_preprocessed_instant,' seconds')

if __name__== "__main__":
    main()

