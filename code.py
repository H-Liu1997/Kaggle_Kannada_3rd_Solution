# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
      
class args:
    # -------- network --------
    batch_size = 500
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 5e-4
    epoch = 100
    factor= 0.1
    step = [10000]
    # -------- data ---------
    augmentation = None
    preprocess = 'normal'
    train_path = '/kaggle/input/Kannada-MNIST/train.csv'
    val_path = '../input/Kannada-MNIST/Dig-MNIST.csv'
    #--------- other --------
    print_fre = 1
    gpu = [0]
    
    
    
class SelfDataSet(torch.utils.data.Dataset):
    def __init__(self,path,augmentation,preprocess):
        self.path = path
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.data = pd.read_csv(self.path)
        self.data2 = self.data.iloc[:,1:].astype('float').values
        self.label = self.data.iloc[:,0].values
    def pre_function(self,ori_img):
        if self.preprocess == 'normal':
            ori_img = ori_img / 255.0 
        else:
            print('error, preprocess type error')
        return ori_img
    def __getitem__(self,index):        
        ori_img = self.data2[index].reshape(-1,28,28)
        label_onehot = np.zeros(10, dtype='float32')
        label_onehot[self.label[index]] =1
        pre_img = self.pre_function(ori_img)
        return pre_img,label_onehot
    def __len__(self):
        ori_csv = pd.read_csv(self.path).values
        return ori_csv.shape[0]
    
    
def loader_factory(data_type,args):
    if data_type == 'train':
        train_dataset = SelfDataSet(args.train_path,args.augmentation,
                                args.preprocess)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                          drop_last=False, num_workers=0,shuffle=True)
        return train_loader
    elif data_type == 'val':
        val_dataset = SelfDataSet(args.val_path,None,
                                args.preprocess)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                          drop_last=False, num_workers=0,shuffle=False)
        return val_loader
    else: print('error,which kind of data you want?')


class Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Block,self).__init__()
                
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(out_channel,out_channel,kernel_size=3),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            #nn.BatchNorm2d(out_channel)
        )
        
    def forward(self,x):
        
        return self.block(x)
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.block1 = Block(1,32)
        self.block2 = Block(32,16)
        self.block3 = Block(16,8)
        #self.batchnorm1 = nn.BatchNorm1d(512)
        #self.batchnorm2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(3872,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,32)
        self.fc5 = nn.Linear(32,10)

    def forward(self,x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0),-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        #x = self.batchnorm1(x)
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        #x = self.batchnorm2(x)
        x = torch.nn.functional.log_softmax(self.fc5(x),dim=-1)        
        return x
       
        
        
class SelfNetwork(torch.nn.Module):
    def __init__(self,args):
        super(SelfNetwork,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,32,3,1,1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32,32,3,1,1)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(2,2)
        
        self.conv3 = torch.nn.Conv2d(32,32,3,1,1)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = torch.nn.Conv2d(32,32,3,1,1)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(2,2)
        
        self.conv5 = torch.nn.Conv2d(32,32,3,1,1)
        self.relu5 = torch.nn.ReLU(inplace=True)
        self.conv6 = torch.nn.Conv2d(32,32,3,1,1)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.conv61 = torch.nn.Conv2d(32,64,3,1,1)
        self.relu61 = torch.nn.ReLU(inplace=True)
        self.conv62 = torch.nn.Conv2d(64,64,3,1,1)
        self.relu62 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool2d(2,2)
        
        self.full1 = torch.nn.Linear(9*64,256)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.full2 = torch.nn.Linear(256,10)
        self.initilization()
        
    def initilization(self):# init checked
        for m in self.modules():
            if isinstance(m,torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias,0.0)
          
    def forward(self,input_):
        x = self.conv1(input_)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv61(x)
        x = self.relu61(x)
        x = self.conv62(x)
        x = self.relu62(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0),-1)
        
        x = self.full1(x)   
        x = self.relu7(x)
        x = torch.nn.functional.log_softmax(self.full2(x),dim=-1)     
        return x


class MyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x,y):        
        loss = -(y*torch.nn.functional.log_softmax(x, dim=1) + 1e-10).cpu().sum(1)
        loss = loss.mean()
        return loss
        
        
def get_optimizer(args,net):
    train_vars = [param for param in net.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(train_vars,lr=args.lr,
                                betas=(args.beta1, args.beta2),
                                eps=1e-08, 
                                weight_decay=args.weight_decay,)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,args.step,
                                                       gamma=args.factor,
                                                      last_epoch=-1)
    return optimizer,lr_schedule
        
def print_terminal(loss,i,epoch,len_data,loss_total):
    str_print = 'epoch: [{0}]{1}/{2}'.format(epoch,i,len_data)
    str_print += ' loss: {loss:.5f}({loss_avg:.5f})'.format(loss=loss,loss_avg=loss_total/(i+1e-10))
    print(str_print)
                    
# body part
def main():
    # init
    train_data = loader_factory('train',args)
    val_data = loader_factory('val',args)
    net = SelfNetwork(args)
    #net = CNN()
    net = torch.nn.DataParallel(net,args.gpu).cuda()
    loss_function = MyLoss()
    optimizer,lr_schedule = get_optimizer(args,net)
    loss_val_min = np.inf
    #loss_function = torch.nn.CrossEntropyLoss()
    # train&&val
    for epoch in range(args.epoch):
        # train
        loss_train_total = 0
        len_train = len(train_data)
        net.train().double()
        for i,(images, label) in enumerate(train_data):
            images = images.cuda()
            label = label.cuda()
            output = net(images)
            loss = loss_function(output,label)
            loss_train_total += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            if i % args.print_fre == 0:
                print_terminal(loss,i,epoch,len_train,loss_train_total)
        # val
        loss_val_total = 0
        len_val = len(val_data)
        net.eval()
        with torch.no_grad():
            for i, (images,label) in enumerate(val_data):
                images = images.cuda()
                label = label.cuda()
                output = net(images)
                loss = loss_function(output,label)
                loss_val_total += loss
                if i%args.print_fre == 0:
                    print_terminal(loss,i,epoch,len_val,loss_val_total)
        loss_val_total /= len(val_data)
        #save
        if loss_val_total < loss_val_min:
            states = { 
               'model_state': net.state_dict(),
               'epoch': epoch + 1,
               'opt_state': optimizer.state_dict(),
               'lr_state': lr_schedule.state_dict()}
            torch.save(states,'best.pth')
    
            
if __name__=='__main__':
    main()
# Any results you write to the current directory are saved as output.
