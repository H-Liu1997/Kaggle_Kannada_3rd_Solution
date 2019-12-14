import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
      
class args(object):
    # -------- network --------
    opt = 'rms'
    batch_size = 256#256#16
    lr = 3e-4
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 1e-4#1e-4
    epoch = 40
    factor= 0.2
    step = [2000,2500,2700]
    patience = 3
    auto_lr_type = 'auto'
    # -------- data ---------
    train_path = '/kaggle/input/Kannada-MNIST/train.csv'
    val_path = '/kaggle/input/Kannada-MNIST/Dig-MNIST.csv'
    test_path = '/kaggle/input/Kannada-MNIST/test.csv'
    result_path = './submission2.csv'
    weight_path = '/kaggle/input/pytorch-kannada-mnist/best.pth'
    weight_save_path = '/kaggle/working/best.pth'
    aug_train = True
    crop_padding = 3
    scale = (0.60,1.40)
    shear = 0.15
    shift = (0.15,0.15)
    angle = (-15,15)
    n_splits = 5
    n_times = 3 
    #--------- other --------
    SEED = 0
    print_fre = 200
    gpu = [0]
    test_number = 0
    test_only = False
    #-------- tricks --------
    #warm_up
    #model_embedding
    #no_bias_decay
    multi_lr = False
    #mix_up
    #cosine lr decay
    #other preprocess
    
def SelfKFold(args):
    '''
    return n train and val dataset.
    '''
    from sklearn.model_selection import KFold
    raw_train_val = pd.read_csv(args.train_path)
    random_train_val = raw_train_val.sample(frac=1).reset_index(drop=True)#need fix np.random
    kfold = KFold(n_splits=args.n_splits,random_state=None)
    train_val_data = raw_train_val.drop('label',axis = 1).values.astype(np.uint8)
    train_val_label = raw_train_val['label'].values.astype(np.uint8)
    print('data shape:', train_val_data.shape)
    print('label shape:', train_val_label.shape)
    train_total = []
    train_label_total = []
    val_total = []
    val_label_total = []
    for _,val_index in kfold.split(train_val_data,train_val_label):
        print('val data index:',val_index[0],val_index[-1])
        train_sub_0 = train_val_data[val_index[-1]+1:]
        train_sub_1 = train_val_data[:val_index[0]]
        val_sub = train_val_data[val_index[0]:val_index[-1]+1]
        train_total.append(np.concatenate((train_sub_0,train_sub_1),axis=0))
        val_total.append(val_sub)

        train_label_sub_0 = train_val_label[val_index[-1]+1:]
        train_label_sub_1 = train_val_label[:val_index[0]]
        val_label_sub = train_val_label[val_index[0]:val_index[-1]+1]
        train_label_total.append(np.concatenate((train_label_sub_0,train_label_sub_1),axis=0))
        val_label_total.append(val_label_sub)
    print('number of dataset:',len(val_total))
    return train_total,train_label_total,val_total,val_label_total
        
    
class SelfDataSet(torch.utils.data.Dataset):
    
    def __init__(self,images, labels = None, augmentation = None,test_data = None,test_label= None): 
        self.augmentation = augmentation
        if  test_data is not None:# for semi-supervise learning
            self.data_image = np.concatenate((images[:,:].reshape(-1,28,28,1),
                                             test_data[:,:].reshape(-1,28,28,1)),0)
            self.label = np.concatenate((np.array(labels.iloc[:],dtype='float32'),
                                         np.array(test_label.iloc[:],dtype='float32')),0)
        elif labels is not None:
            self.data_image = images.reshape(-1,28,28,1)
          # values useless
            self.label = labels
           
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
            #print(self.label.shape)
            label_onehot[self.label[index]] = 1
            #print('here4')
            return pre_img,label_onehot
        
    def __len__(self): 
        return self.data_image.shape[0]

    
def loader_factory(data_type,args,train_image=None,train_label=None,val_image=None,val_label=None,TTA=0):
    
    # ------- augment portion -------- #
    if args.aug_train == False:
        transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
    else:
        transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(args.angle, translate=args.shift, scale=args.scale,
                                    shear=args.shear, resample=False, fillcolor=0),
            #transforms.RandomResizedCrop(28, scale=args.scale, ratio=(1, 1), interpolation=2),
            #transforms.RandomRotation(args.angle),
            transforms.ToTensor(),
        ])
    transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    if TTA == 0:
        transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), ])
    elif TTA == 1:
        transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), ])
    else :
        print('TTA type error')
       
    # ------- data portion -------- #
    if data_type == 'train_total_data':
        ori_data = pd.read_csv(args.train_path)
        ori_label = ori_data['label']
        ori_data.drop('label',axis=1,inplace=True)
        train_dataset = SelfDataSet(ori_data,ori_label,transforms_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                          drop_last=False, num_workers=0,shuffle=True)
        return train_loader
    
    elif data_type == 'train&&val':
        train_dataset = SelfDataSet(train_image,train_label,transforms_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   drop_last=False, num_workers=0,shuffle=True)
        val_dataset = SelfDataSet(val_image,val_label,transforms_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                      drop_last=False, num_workers=0,shuffle=False)   
        return train_loader,val_loader

    elif data_type == 'test':
         ori_data = pd.read_csv(args.test_path)
         ori_data.drop('id',axis=1,inplace=True)   
        
         test_dataset = SelfDataSet(ori_data,None,transforms_test)
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
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim,channels,3,1,1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        self.conv2 = nn.Sequential(nn.Conv2d(channels,channels,3,1,1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        self.conv3 = nn.Sequential(nn.Conv2d(channels,channels,3,1,1),
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
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim,channels,3,1,1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(channels))
        self.conv2 = nn.Sequential(nn.Conv2d(channels,channels,3,1,1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(channels))
        # pass self init
    def forward(self,input_):
        x0 = self.conv1(input_)
        x1 = self.conv2(x0)
        x_out = torch.cat((x0,x1),1)
        return x_out

class SelfDenseNet2(torch.nn.Module):
    def __init__(self,args):
        super(SelfDenseNet2,self).__init__()
        self.net_cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(64,momentum=0.01),
            nn.Conv2d(64,64,3,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(64,momentum=0.01),
            nn.Conv2d(64,64,5,1,2),
            nn.ReLU(),
            nn.BatchNorm2d(64,momentum=0.01),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64,128,3,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(128,momentum=0.01),
            nn.Conv2d(128,128,3,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(128,momentum=0.01),
            nn.Conv2d(128,128,5,1,2),
            nn.ReLU(),
            nn.BatchNorm2d(128,momentum=0.01),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128,256,3,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(256,momentum=0.01),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            
            nn.Flatten(),
            nn.Linear(256,256),
            nn.BatchNorm1d(256,momentum=0.01),
            nn.Linear(256,128),
            nn.BatchNorm1d(128,momentum=0.01),
            nn.Linear(128,10)
        )
        self.initilization()
        
    def initilization(self):# init checked
        for m in self.net_cnn:
            if isinstance(m,torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                print('init weight success')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias,0.0)
                    print('init bias success')
        
    def forward(self,input_):
        x = self.net_cnn(input_)       
        x = torch.nn.functional.softmax(x,dim=-1)     
        return x    
    
class SelfNetwork2(torch.nn.Module):
    def __init__(self,args):
        super(SelfNetwork2,self).__init__()
        self.net_cnn = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,1,1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            
            DenseBlock2(32,32),
            nn.Dropout(0.4),
            DenseBlock2(64,32),
            nn.MaxPool2d(2,2),
            
            DenseBlock3(64,64),
            DenseBlock3(64*3,64),
            nn.Dropout(0.4),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64*3,128,3,1,0),
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
        x = torch.nn.functional.softmax(self.fc1(x),dim=-1)     
        return x


class MyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x,y):
        #TODO: here add focus loss
        loss = -(y*torch.log(x)).cpu().sum(1)
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
    elif args.opt == 'rms':
        optimizer = torch.optim.RMSprop(train_vars, lr=args.lr, 
                                        alpha=args.beta1, eps=1e-07, weight_decay=args.weight_decay, 
                                        momentum=0, centered=False)
    else: print('optimizer type error')
    if args.auto_lr_type == 'ms': 
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,args.step,
                                                       gamma=args.factor,
                                                      last_epoch=-1)
    else:
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=args.factor, patience=args.patience, verbose=True, threshold=0.0001,
                threshold_mode='rel', cooldown=3, min_lr=1e-6, eps=1e-08)
    return optimizer,lr_schedule

        
def print_terminal(loss,i,epoch,len_data,loss_total):
    str_print = 'epoch: [{0}]{1}/{2}'.format(epoch,i,len_data)
    str_print += ' loss: {loss:.5f}({loss_avg:.5f})'.format(loss=loss,loss_avg=loss_total/(i+1e-10))
    print(str_print)
          

def test_(net,weight_path,test_data,weight_save_path):
    state = torch.load(weight_path)
    torch.save(state,weight_save_path)
    weight_state = state['model_state']
    net.load_state_dict(weight_state)
    print('load weight success')
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
    
# body part
def main(args): 
    random.seed(args.SEED)# add this line when using torchvision's transforms
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.SEED)
    #torch.backends.cudnn.benchmark = False # default is False
    #torch.cuda.manual_seed_all(SEED)      # no influence
     
    # init
    train_total,train_label_total,val_total,val_label_total = SelfKFold(args)
    print(len(train_total))
    train_accuracy = np.zeros((args.n_times))
    val_accuracy___ = np.zeros((args.n_times))
    train_curve = []
    val_curve = []
    
    for n_time in range(args.n_times):        
        train_data,val_data = loader_factory('train&&val',args,train_total[n_time],
                                            train_label_total[n_time],val_total[n_time],val_label_total[n_time])   
        net = SelfDenseNet2(args)
        print(net)
        #net = SelfDenseNet(args)
        #net = CNN()
        net = torch.nn.DataParallel(net,args.gpu).cuda()
        loss_function = MyLoss()
        optimizer,lr_schedule = get_optimizer(args,net)
        loss_val_min = np.inf
        # train&&val
        train_curve_single = []
        val_curve_single = []
        
        for epoch in range(args.epoch):
            # train
            if args.test_only:
                break
            loss_train_total = 0
            len_train = len(train_data)
            net.train()
            correct = 0
            img_count = 0
            accuracy = 0
            for i,(images, label) in enumerate(train_data):
                images = images.cuda()
                label = label.cuda()
                output = net(images)

                pred = output.data.max(1 , keepdim=True)[1]            
                correct += pred.eq(label.max(1, keepdim=True)[1].data.view_as(pred)).sum().cpu().numpy()
                img_count += images.shape[0]

                loss = loss_function(output,label)
                loss_train_total += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.auto_lr_type == 'ms':
                    lr_schedule.step()
                if i % args.print_fre == 0:
                    print_terminal(loss,i,epoch,len_train,loss_train_total)
            
            accuracy = correct / img_count
            train_curve_single.append((loss_train_total / len_train,accuracy))
            print(accuracy)
            # val
            
            if epoch == args.epoch-1:
                states = { 
                   'model_state': net.state_dict(),
                   'epoch': epoch + 1,
                   'opt_state': optimizer.state_dict(),
                   'lr_state': lr_schedule.state_dict()}
                torch.save(states,args.weight_save_path)

            loss_val_total = 0
            net.eval() 
            val_img_count = 0
            val_correct = 0
            val_accuracy = 0
            len_val = len(val_data)
            with torch.no_grad():
                for i, (images,label) in enumerate(val_data):
                    images = images.cuda()
                    label = label.cuda()
                    output = net(images)
                    loss = loss_function(output,label)
                    loss_val_total += loss

                    pred = output.data.max(1 , keepdim=True)[1]
                    val_correct += pred.eq(label.max(1, keepdim=True)[1].data.view_as(pred)).sum().cpu().numpy()
                    val_img_count += images.shape[0]           
                    if i%args.print_fre == 0:
                        print_terminal(loss,i,epoch,len_val,loss_val_total)
                val_accuracy = val_correct / val_img_count
                print(val_accuracy)
            loss_val_total /= len_val
            val_curve_single.append((loss_val_total,val_accuracy))
            if args.auto_lr_type != 'ms':
                lr_schedule.step(loss_train_total)
                
        train_accuracy[n_time]=accuracy
        val_accuracy___[n_time]=val_accuracy
        train_curve.append(train_curve_single)
        val_curve.append(val_curve_single)
        
    print(np.mean(train_accuracy))
    print(np.mean(val_accuracy___))
        #save
        
        #if loss_val_total < loss_val_min:
        
    if args.test_only:
        test_(net,args.weight_path,test_data,args.weight_save_path)
    
    
    

    
    
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
    args = args()
    main(args)
    #args.weight_save_path = '/kaggle/working/best_1.pth'
    #args.split = 0  
   # main(args)
#     submission = pd.read_csv('/kaggle/input/pytorch-kannada-mnist/submission0.csv')
#     submission.to_csv('/kaggle/working/submission.csv', index=False)
#     submission.head()
    
# Any results you write to the current directory are saved as output
