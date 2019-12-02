# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
      
class args:
    # -------- network --------
    opt = 'adam'
    batch_size = 256#16
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 5e-4
    epoch = 35
    factor= 0.100
    step = [20000,30000]
    # -------- data ---------
    train_path = '/kaggle/input/Kannada-MNIST/train.csv'
    val_path = '/kaggle/input/Kannada-MNIST/Dig-MNIST.csv'
    test_path = '/kaggle/input/Kannada-MNIST/test.csv'
    result_path = './submission2.csv'
    crop_padding = 3
    scale = (0.9,1.1)
    angle = [-15,15]  
    #--------- other --------
    print_fre = 40
    gpu = [0]
    #-------- tricks --------
    #warm_up
    #model_embedding
    #no_bias_decay
    multi_lr = False
    #mix_up
    #cosine lr decay
    #other preprocess
    
        
class SelfDataSet(torch.utils.data.Dataset):
    
    def __init__(self,images, labels = None, augmentation = None,test_data = None,test_label= None): 
        self.augmentation = augmentation
        if  test_data is not None:# for semi-supervise learning
            self.data_image = np.concatenate((np.array(images.iloc[:,:],dtype='uint8').reshape(-1,28,28,1),
                                              np.array(test_data.iloc[:,:],dtype='uint8').reshape(-1,28,28,1)),0)
            self.label = np.concatenate((np.array(labels.iloc[:],dtype='float32'),
                                         np.array(test_label.iloc[:],dtype='float32')),0)
        elif labels is not None:
            self.data_image = np.array(images.iloc[:,:],dtype='uint8').reshape(-1,28,28,1)
          # values useless
            self.label = np.array(labels.iloc[:])
           
        else:
            self.data_image = np.array(images.iloc[:,:],dtype='uint8').reshape(-1,28,28,1)
            self.label = None
                                  
    def __getitem__(self,index):
        #print('here')
        ori_img = self.data_image[index] 
        #print('here2')
        pre_img = self.augmentation(ori_img)    
        if self.label is None:
            return pre_img
        else:
            label_onehot = np.zeros(10, dtype='float32')
            #print('here3')
           # print(self.label.shape)
            label_onehot[self.label[index]] = 1
            #print('here4')
            return pre_img,label_onehot
        
    def __len__(self): 
        return self.data_image.shape[0]

    
def loader_factory(data_type,args):
    
    transforms_train = transforms.Compose([
        transforms.ToPILImage(),      
        #transforms.RandomResizedCrop(28, scale=args.scale, ratio=(1, 1), interpolation=2),
        #transforms.RandomRotation(args.angle),
        transforms.ToTensor(),
    ])
    transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    if data_type == 'train&&val':
        ori_data = pd.read_csv(args.train_path)
        ori_label = ori_data['label']
        ori_data.drop('label',axis=1,inplace=True)
        
        train_image, val_image, train_label, val_label = train_test_split(ori_data,ori_label,
                                                          stratify=ori_label, random_state = 0,
                                                          test_size = 0.005)
        train_dataset = SelfDataSet(train_image,train_label,transforms_train)
        
        val_dataset = SelfDataSet(val_image,val_label,transforms_val)
       
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                          drop_last=False, num_workers=0,shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                          drop_last=False, num_workers=0,shuffle=False)   
        return train_loader,val_loader

    elif data_type == 'test':
         ori_data = pd.read_csv(args.test_path)
         ori_data.drop('id',axis=1,inplace=True)
    
        
         test_dataset = SelfDataSet(ori_data,None,transforms_val)
         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                          drop_last=False, num_workers=0,shuffle=False)
         return test_loader
        
    elif data_type == 'test_train':
         ori_data = pd.read_csv(args.train_path)
         ori_label = ori_data['label']
         ori_data.drop('label',axis=1,inplace=True)
         test_data = pd.read_csv(args.test_path)
         test_data.drop('label',axis=1,inplace=True)
         test_label = pd.read_csv(args.results_path)    
         test_label.drop('id',axis=1,inplace=True)
            
         train_test_dataset = SelfDataSet(ori_data,ori_label,transforms_train,test_data,test_label)
         train_test_loader = torch.utils.data.DataLoader(train_test_dataset, batch_size=args.batch_size,
                                          drop_last=False, num_workers=0,shuffle=True)
         return train_test_loader
        
    else: print('error,which kind of data you want?')


        
class DenseBlock3(torch.nn.Module):
    def __init__(self,in_dim,channels):
        super(DenseBlock3,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim,channels,3,1,0),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        self.conv2 = nn.Sequential(nn.Conv2d(channels,channels,3,1,0),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        self.conv3 = nn.Sequential(nn.Conv2d(channels,channels,3,1,0),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        # pass self init
    def forward(self,input_):
        x0 = self.conv1(input_)
        x1 = self.conv2(x0)
        x2 = self.conv3(x1)
        x_out = torch.cat((x0,x1,x2),1)
        return x_out
    
class DenseBlock2(torch.nn.Module):
    def __init__(self,in_dim,channels):
        super(DenseBlock2,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim,channels,3,1,0),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(channels))
        self.conv2 = nn.Sequential(nn.Conv2d(channels,channels,3,1,0),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(channels))
        # pass self init
    def forward(self,input_):
        x0 = self.conv1(input_)
        x1 = self.conv2(x0)
        x_out = torch.cat((x0,x1),1)
        return x_out

class SelfDenseNet(torch.nn.Module):
    def __init__(self,args):
        super(SelfDenseNet,self).__init__()
        self.net_cnn = nn.Sequential(
            nn.Conv2d(1,32,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            
            DenseBlock2(32,32),
            nn.Dropout(0.4),
            DenseBlock2(96,32),
            
            DenseBlock3(96,64),
            DenseBlock3(64*3,64),
            nn.Dropout(0.4),
            
            nn.Conv2d(64*3,128,4,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.4),   
        )
        self.fc1 = nn.Linear(128,10)
        
    def forward(self,input_):
        x = self.net_cnn(input_)       
        x = x.view(x.size(0),-1)
        x = torch.nn.functional.log_softmax(self.fc1(x),dim=-1)     
        return x
        
class SelfNetwork(torch.nn.Module):
    def __init__(self,args):
        super(SelfNetwork,self).__init__()
        self.net_cnn = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,1,1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32,32,5,1,2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
            nn.Conv2d(32,64,5,1,2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            
            
            nn.Conv2d(64,64,7,1,3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,7,1,3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,128,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.4),   
        )
        self.fc1 = nn.Linear(128,10)
        '''self.net_cnn = nn.Sequential(
            nn.Conv2d(1,32,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32,32,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
            nn.Conv2d(32,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            
            
            nn.Conv2d(64,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            nn.Conv2d(64,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            
            nn.Conv2d(64,128,4,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.4),   
        )
        self.fc1 = nn.Linear(128,10)'''
        #self.initilization()
        
    def initilization(self):# init checked
        for m in self.modules():
            if isinstance(m,torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias,0.0)
          
    def forward(self,input_):
        x = self.net_cnn(input_)       
        x = x.view(x.size(0),-1)
        x = torch.nn.functional.log_softmax(self.fc1(x),dim=-1)     
        return x


class MyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x,y):
        #TODO: here add focus loss
        loss = -(y*torch.nn.functional.log_softmax(x, dim=1) + 1e-10).cpu().sum(1)
        loss = loss.mean()
        return loss
        
        
def get_optimizer(args,net):
    
    if args.multi_lr:
        decay_1, no_decay_2 = [],[]
        for name, param in net.named_parameters():           
            if name.endswith(".bias"):
                print(name[7:],"using no_decay_2")
                no_decay_2.append(param)
            else:
                decay_1.append(param)
        train_vars = [{'params': decay_1, 'lr': args.lr, 'weight_decay':args.weight_decay},
                     {'params': no_decay_2, 'lr': args.lr, 'weight_decay':0},]
    else:
        train_vars = [param for param in net.parameters() if param.requires_grad]
                
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(train_vars,lr=args.lr,
                                betas=(args.beta1, args.beta2),
                                eps=1e-08, 
                                weight_decay=args.weight_decay,)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(train_vars,lr=args.lr,
                                momentum=args.beta1,
                                weight_decay=args.weight_decay,)
    else: print('optimizer type error')
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
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    # init
    train_data,val_data = loader_factory('train&&val',args)
    test_data = loader_factory('test',args)
    
    net = SelfNetwork(args)
    #net = SelfDenseNet(args)
    #net = CNN()
    net = torch.nn.DataParallel(net,args.gpu).cuda()
    loss_function = MyLoss()
    optimizer,lr_schedule = get_optimizer(args,net)
    loss_val_min = np.inf
    # train&&val
    for epoch in range(args.epoch):
        # train
        loss_train_total = 0
        len_train = len(train_data)
        net.train()
        correct = 0
        accuracy = 0
        for i,(images, label) in enumerate(train_data):
            images = images.cuda()
            label = label.cuda()
            output = net(images)
            
            pred = output.data.max(1 , keepdim=True)[1]            
            correct += pred.eq(label.max(1, keepdim=True)[1].data.view_as(pred)).sum().cpu().numpy()
                    
            loss = loss_function(output,label)
            loss_train_total += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            if i % args.print_fre == 0:
                print_terminal(loss,i,epoch,len_train,loss_train_total)
        accuracy = correct / (len_train*args.batch_size)
        print(accuracy)
        # val
        loss_val_total = 0
        len_val = len(val_data)
        net.eval()  
        val_correct = 0
        val_accuracy = 0
        with torch.no_grad():
            for i, (images,label) in enumerate(val_data):
                images = images.cuda()
                label = label.cuda()
                output = net(images)
                loss = loss_function(output,label)
                loss_val_total += loss
                            
                pred = output.data.max(1 , keepdim=True)[1]
                val_correct += pred.eq(label.max(1, keepdim=True)[1].data.view_as(pred)).sum().cpu().numpy()
                           
                if i%args.print_fre == 0:
                    print_terminal(loss,i,epoch,len_val,loss_val_total)
            val_accuracy = val_correct / (len_val*args.batch_size)
            print(val_accuracy)
        loss_val_total /= len(val_data)
        #save
        if loss_val_total < loss_val_min:
            states = { 
               'model_state': net.state_dict(),
               'epoch': epoch + 1,
               'opt_state': optimizer.state_dict(),
               'lr_state': lr_schedule.state_dict()}
            torch.save(states,'best.pth')
    #test
    predictions = []
    with torch.no_grad():
        for i, images in enumerate(test_data):
            images = images.cuda()
            output = net(images).max(dim=1)[1]
            predictions += list(output.data.cpu().numpy())
    submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
    submission['label'] = predictions
    submission.to_csv('submission.csv', index=False)
    submission.head()
    
    
#     traintest_data = loader_factory('test_train',args)
#     net2 = SelfNetwork(args)
#     #net = CNN()
#     net2 = torch.nn.DataParallel(net2,args.gpu).cuda()
#     optimizer2,lr_schedule2 = get_optimizer(args,net2)
#     loss_val_min = np.inf
#     print('start')
#     for epoch in range(args.epoch):
#         # train
#         loss_train_total = 0
#         len_train = len(traintest_data)
#         print(len_train)
#         net2.train()
#         correct = 0
#         accuracy = 0
#         for i,(images, label) in enumerate(traintest_data):
#             images = images.cuda()
#             label = label.cuda()
#             output = net2(images)
            
#             pred = output.data.max(1 , keepdim=True)[1]            
#             correct += pred.eq(label.max(1, keepdim=True)[1].data.view_as(pred)).sum().cpu().numpy()
                    
#             loss = loss_function(output,label)
#             loss_train_total += loss
#             optimizer2.zero_grad()
#             loss.backward()
#             optimizer2.step()
#             lr_schedule2.step()
#             if i % args.print_fre == 0:
#                 print_terminal(loss,i,epoch,len_train,loss_train_total)
#         accuracy = correct / (len_train*args.batch_size)
#         print(accuracy)
#         # val
#         loss_val_total = 0
#         len_val = len(val_data)
#         net2.eval()  
#         val_correct = 0
#         val_accuracy = 0
#         with torch.no_grad():
#             for i, (images,label) in enumerate(val_data):
#                 images = images.cuda()
#                 label = label.cuda()
#                 output = net2(images)
#                 loss = loss_function(output,label)
#                 loss_val_total += loss
                            
#                 pred = output.data.max(1 , keepdim=True)[1]
#                 val_correct += pred.eq(label.max(1, keepdim=True)[1].data.view_as(pred)).sum().cpu().numpy()
                           
#                 if i%args.print_fre == 0:
#                     print_terminal(loss,i,epoch,len_val,loss_val_total)
#             val_accuracy = val_correct / (len_val*args.batch_size)
#             print(val_accuracy)
#         loss_val_total /= len(val_data)
        
#     predictions = []
#     with torch.no_grad():
#         for i, images in enumerate(test_data):
#             images = images.cuda()
#             output = net2(images).max(dim=1)[1]
#             predictions += list(output.data.cpu().numpy())
#     submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
#     submission['label'] = predictions
#     submission.to_csv('submission.csv', index=False)
#     submission.head()
            
            
if __name__=='__main__':
    main()
# Any results you write to the current directory are saved as output.
