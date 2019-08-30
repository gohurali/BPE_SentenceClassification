__author__ = 'Gohur Ali'
import numpy as np
import os               # FileSystem Access
import yaml             # Config File Access
from tqdm import tqdm   # Progress Visualization
import time
import argparse
import json
import sys
import pickle
import re
import codecs
from bpemb import BPEmb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.architectures import ShallowCNN
from utils.preprocessing import DataPrepper

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(str(device) == 'cuda'):
    print("Device state:\t", device)
    print("Device index:\t",torch.cuda.current_device())
    print("Current device:\t", torch.cuda.get_device_name(device))
cfg = yaml.safe_load(open('config.yaml'))


class Trainer:
    def __init__(self,config={},DataPrepper=None):
        self.cfg = config
        self.dataprepper = DataPrepper
        self.bpe_model, self.embeddings = self.open_bpe_vectors()

        self.x_train = self.bpe_model.encode_ids(self.dataprepper.x_train)
        self.x_test = self.bpe_model.encode_ids(self.dataprepper.x_test)
        self.x_train = pad_sequences(sequences=self.x_train,maxlen=self.cfg['pad_limit'])
        self.x_test = pad_sequences(sequences=self.x_test, maxlen=self.cfg['pad_limit'])         
        self.y_train = self.dataprepper.y_train.reshape((self.dataprepper.y_train.shape[0],1))
        self.y_test = self.dataprepper.y_test.reshape((self.dataprepper.y_test.shape[0],1))

        self.train_idx_labels = self.y_train
        self.test_idx_labels = self.y_test

        if(self.cfg['if_softmax']):
            self.y_train = self.to_categorical(self.y_train, self.cfg['num_classes'])
            self.y_test = self.to_categorical(self.y_test, self.cfg['num_classes'])
        
        print('Train data size: x_train = {',self.x_train.shape,'} -- y_train = {',self.y_train.shape,'}')
        print('Test data size: x_test = {',self.x_test.shape,'} -- y_test = {',self.y_test.shape,'}')

        self.train_dataloader,self.test_dataloader = self.create_dataloaders(
            train_data=(self.x_train,self.y_train),
            test_data=(self.x_test,self.y_test)
            )
        pass
      
    def create_dataloader(self, features, labels):
        print('-- Batch size ',self.cfg['batch_size'],'--')
        dataset = torch.utils.data.TensorDataset(features, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=True)
        return data_loader
    
    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]              
        
    def open_bpe_vectors(self):
        en_model = BPEmb(lang='en',vs=200000,dim=300)
        return en_model, en_model.vectors
        
    
    def build_embedding_table(self, mapping):
        table = np.zeros((len(mapping), self.cfg['embedding_dim']))
        for word, value in mapping.items():
            if(value[1] is not None):
                table[value[0]] = value[1]
        return table
    
    def split_data(self,examples,labels):
        if(self.use_default_split == False):
            cfg_split_ratio = self.cfg['train_test_split_ratio']
            x_train, x_test, y_train, y_test = train_test_split(self.examples, self.labels, test_size=cfg_split_ratio, random_state=1000)
            return x_train,x_test,y_train,y_test

    def create_dataloaders(self,train_data,test_data):
        x_train = train_data[0]
        y_train = train_data[1]

        x_test = test_data[0]
        y_test = test_data[1]

        if(str(device) == 'cuda'):
            x_train = torch.tensor(x_train).to(device)#.cuda()
            y_train = torch.tensor(y_train,dtype=torch.long).to(device)#.cuda()
            x_test = torch.tensor(x_test).to(device)#.cuda()
            y_test = torch.tensor(y_test,dtype=torch.long).to(device)#.cuda()
        else:
            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train,dtype=torch.long)
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test,dtype=torch.long)

        train_dataloader = self.create_dataloader(features=x_train, labels=y_train)
        test_dataloader = self.create_dataloader(features=x_test, labels=y_test)
        return train_dataloader,test_dataloader

    
    def build_model(self, embeddings):
        return ShallowCNN(self.cfg,embeddings)
    
    def train(self,train_data):

        epochs = 30 # self.cfg['epochs']
        learning_rate = 0.0001 #self.cfg['learning_rate'])
        
        # -- Create Model --
        self.model = self.build_model(torch.tensor(self.embeddings))
        print(self.model)

        # -- Model to CUDA GPU --
        if( str(device) == 'cuda'):
            print('Sending model to',torch.cuda.get_device_name(device),' GPU')
            #model = model.cuda()
            self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
        #                                             step_size=50,
        #                                             gamma=0.1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1,patience=5,
            verbose=True,threshold=0.0001, threshold_mode='rel', 
            cooldown=0,min_lr=0,eps=1e-08
            )
        self.loss_function = None
        
        if(self.cfg['if_softmax']):
            #self.loss_function = torch.nn.CrossEntropyLoss()
            self.loss_function = torch.nn.NLLLoss()
        else:
            self.loss_function = torch.nn.BCELoss()

        
        accuracy = 0
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            loss = 0
            correct = 0
            for i , (examples, labels) in tqdm(enumerate(train_data)):
                if( self.cfg['if_softmax']):
                    labels_n = labels.cpu().numpy()
                    labels_idx = np.argwhere(labels_n >0)
                    labels_idx = labels_idx.T
                    labels_idx = np.delete(labels_idx,0,0).T
                    labels_idx = np.squeeze(labels_idx,1)
                    labels_idx = torch.tensor(labels_idx,dtype=torch.int)
                    #print(labels_idx)

                    # Transfer to GPU
                    if(str(device) == 'cuda'):
                        examples = examples.to(device)
                        labels = labels.to(device)
                        labels_idx = labels_idx.to(device)
                    
                    self.model.zero_grad()

                    predictions = self.model(examples.long())
                    loss = self.loss_function(predictions,labels_idx.long())
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                else:
                    if(str(device) == 'cuda'):
                        examples = examples.to(device)
                        labels = labels.to(device)
                    self.model.zero_grad()

                    predictions = self.model(examples.long())
                    if(str(device) == 'cuda'):
                        predictions = predictions.to(device)
                    loss = self.loss_function(predictions.float(),labels.float())

                    #preds = np.round(predictions.float().cpu().detach())
                    preds = torch.round(predictions.float().detach())
                    labels = labels.float().detach()
                    #labels = labels.float().cpu().detach()
                    correct += (preds == labels).sum()
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                #break
            scheduler.step(total_loss) 
            losses.append(total_loss)
            #break
            accuracy = correct.float()/self.x_train.shape[0] * 100
            #print('Epoch {} ----> loss={} accuracy={}'.format(epoch,total_loss,accuracy))
            print(f'Epoch {epoch} ----> loss={total_loss:.5f} accuracy={accuracy:.5f}')
            if(torch.cuda.memory_allocated(device)/1000000000 > 10.5):
                print('GPU Memory Occupied: {}'.format(torch.cuda.memory_allocated(device)/1000000000))
            #print('Epoch {} Learning_Rate{} ----> loss={}'.format(epoch,scheduler.get_lr(),total_loss))
            print('=====================================================================================================')
        return self.model, self.loss_function, losses
    
    def save_model(self):
        torch.save(self.model.state_dict(),self.cfg['save_location']+'pt_'+self.cfg['dataset']+'.pt')
    
    def checkpoint_model(self):
        chkpt = {
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.loss_function.state_dict()
        }
        torch.save(chkpt,self.cfg['save_location']+'pt_'+self.cfg['dataset']+'.chkpt')
    
    def test_validate(self,debug=False,model=None,test_data=[],loss_fn=None):
        if(model == None and loss_fn == None):
            if(self.cfg['if_softmax']):
                #self.loss_function = torch.nn.CrossEntropyLoss()
                loss_fn = torch.nn.NLLLoss()
            else:
                loss_fn = torch.nn.BCELoss()
            self.model = ShallowCNN(self.cfg,torch.tensor(self.embeddings))
            #self.model.load_state_dict(torch.load(self.cfg['save_location']+'pt_'+self.cfg['dataset']+'.chkpt'))
            checkpoint = torch.load(self.cfg['save_location']+'pt_'+self.cfg['dataset']+'.chkpt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            loss_fn.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()

            if(str(device) == 'cuda'):
                self.model.to(device)
        test_loss = 0
        correct = 0
        all_predictions = []
        for idx,(examples, labels) in enumerate(test_data):
            if( self.cfg['if_softmax']):
                labels_n = labels.cpu().numpy()
                labels_idx = np.argwhere(labels_n >0)
                labels_idx = labels_idx.T
                labels_idx = np.delete(labels_idx,0,0).T
                labels_idx = np.squeeze(labels_idx,1)
                labels_idx = torch.tensor(labels_idx,dtype=torch.int)
                if(str(device) == 'cuda'):
                    examples = examples.to(device)
                    labels = labels.to(device)
                    labels_idx = labels_idx.to(device)

                outputs = self.model.forward(examples.long())

                preds = []
                for pred in outputs:
                    #preds.append((torch.max(pred).detach(),np.argmax(pred.cpu().detach().numpy())))
                    preds.append(np.argmax(pred.cpu().detach().numpy()))
                preds = torch.tensor(preds,dtype=torch.int).to(device)
                
                all_predictions.append(outputs)
                loss = loss_fn(outputs, labels_idx.long())
                test_loss += loss.item()

                correct += (preds == labels_idx).sum() 

                if(debug):
                    for ex,label,label_idx,pred,pred_idx in zip(examples,labels,labels_idx,outputs,preds):
                        print('{}: actual = {} ---> pred = {}'.format(idx,label_idx.item(),pred_idx.item()))
            else:
                if(str(device) == 'cuda'):
                    examples = examples.to(device)
                    labels = labels.to(device)
                outputs = self.model.forward(examples.long())
                all_predictions.append(outputs)
                loss = loss_fn(outputs.float(), labels.float())
                test_loss += loss.item()


                preds = np.round(outputs.float().cpu().detach())
                labels = labels.float().cpu().detach()
                correct += (preds == labels).sum()

                if(debug):
                   for ex,label,pred in zip(examples,labels,preds):
                       print('{}: actual = {} ---> pred = {}'.format(idx,label.item(),pred.item()))
            
            # print('correct = ',correct)
            #accuracy = correct.float()/64 * 100
      
        accuracy = correct.float()/self.x_test.shape[0] * 100
        return test_loss, accuracy, all_predictions

def main():
    dp = DataPrepper(config=cfg,dataset=cfg['dataset'])
    trainer = Trainer(config=cfg,DataPrepper=dp)
    
    model,criterion,losses = trainer.train(train_data=trainer.train_dataloader)
    test_loss,acc,preds = trainer.test_validate(debug=True,model=trainer.model,test_data=trainer.test_dataloader,loss_fn=criterion)
    print('Test Accuracy = {}%'.format(acc))

if __name__ == '__main__':
    main()