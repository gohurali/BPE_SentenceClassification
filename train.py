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

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(str(device) == 'cuda'):
    print("Device state:\t", device)
    print("Device index:\t",torch.cuda.current_device())
    print("Current device:\t", torch.cuda.get_device_name(device))

class Trainer:
    def __init__(self):
        self.cfg = yaml.safe_load(open('config.yaml'))
        # Used pre-determined data split
        self.use_default_split = True
        
        if(self.use_default_split):
            self.x_train,self.y_train,self.x_test,self.y_test = self.get_trec_dataset(self.cfg['train_data_location'], use_default_split=self.use_default_split)
                      
            self.bpe_model, self.embeddings = self.open_bpe_vectors()

            self.x_train = self.bpe_model.encode_ids(self.x_train)
            self.x_test = self.bpe_model.encode_ids(self.x_test)
            self.x_train = pad_sequences(sequences=self.x_train,maxlen=self.cfg['pad_limit'])
            self.x_test = pad_sequences(sequences=self.x_test, maxlen=self.cfg['pad_limit'])         
                        
            self.train_idx_labels = self.y_train
            self.test_idx_labels = self.y_test
            
        if(self.cfg['if_softmax']):
            self.y_train = self.to_categorical(self.y_train, self.cfg['num_classes'])
            self.y_test = self.to_categorical(self.y_test, self.cfg['num_classes'])
        else:
            self.examples, self.labels = self.get_trec_dataset(self.cfg['train_data_location'], use_default_split=self.use_default_split)
            self.examples = self.sequence_examples(self.examples)
        
        print('Train data size: x_train = {',self.x_train.shape,'} -- y_train = {',self.y_train.shape,'}')
        print('Test data size: x_test = {',self.x_test.shape,'} -- y_test = {',self.y_test.shape,'}')

        self.train_dataloader,self.test_dataloader = self.create_dataloaders(train_data=(self.x_train,self.y_train),
                                                                             test_data=(self.x_test,self.y_test)
                                                                             )
        pass
        
    def sequence_examples(self, dataset):
        sequenced_dataset = []
        for example in tqdm(dataset):
            sequenced_sentence = []
            words = example.split()
            for word in words:
                if(word in self.w2e.keys()):
                    idx = self.w2e[word][0]
                    sequenced_sentence.append(idx)
                else:
                    idx = self.w2e['_unk'][0]
                    sequenced_sentence.append(idx)
            sequenced_dataset.append(sequenced_sentence)
        return sequenced_dataset
      
    def create_dataloader(self, features, labels):
        print('-- Batch size ',self.cfg['batch_size'],'--')
        dataset = torch.utils.data.TensorDataset(features, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=True)
        return data_loader
    
    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]              
        
    def get_trec_dataset(self, train_data_location, use_default_split=False):
        """Open and prepare the subjectivity dataset. Using
        Regular expressions to clean the sentences.
        
        Args:
            train_data_location - location of the data, specified in config
        Return:
            dataset - ndarray of each example
            labels - array of binary labels
        """
        
        if(use_default_split == False):
            dataset = []
            labels = []
            for f in os.listdir(train_data_location):
                print(f)
                if(f == 'trec_5000_train.txt'):
                    # Subjective Data
                    with open(train_data_location + f, encoding = "ISO-8859-1") as subj_file:
                        for line in subj_file:
                            split_line = line.split(':')
                            ques_class = split_line[0]
                            question = split_line[1]
                            pattern = "[^a-zA-Z.' ]"
                            cleaned_line = re.sub(pattern,' ',question)
                            cleaned_line = cleaned_line.lower()
                            dataset.append(cleaned_line)
                            if(ques_class == 'NUM'):
                                labels.append(0)
                            elif(ques_class == 'DESC'):
                                labels.append(1)
                            elif(ques_class == 'ENTY'):
                                labels.append(2)
                            elif(ques_class == 'HUM'):
                                labels.append(3)
                            elif(ques_class == 'ABBR'):
                                labels.append(4)
                            elif(ques_class == 'LOC'):
                                labels.append(5)
                elif(f == 'trec_test.txt'):
                    # Objective Data
                    with open(train_data_location + f, encoding = "ISO-8859-1") as obj_file:
                        for line in obj_file:
                            split_line = line.split(': ')
                            ques_class = split_line[0]
                            question = split_line[1]
                            pattern = "[^a-zA-Z.' ]"
                            cleaned_line = re.sub(pattern,' ',question)
                            cleaned_line = cleaned_line.lower()
                            dataset.append(cleaned_line)
                            if(ques_class == 'NUM'):
                                labels.append(0)
                            elif(ques_class == 'DESC'):
                                labels.append(1)
                            elif(ques_class == 'ENTY'):
                                labels.append(2)
                            elif(ques_class == 'HUM'):
                                labels.append(3)
                            elif(ques_class == 'ABBR'):
                                labels.append(4)
                            elif(ques_class == 'LOC'):
                                labels.append(5)
            return np.array(dataset), np.array(labels)
        elif(use_default_split==True):
            x_train = []
            x_test = []
            y_train = []
            y_test = []
            for f in os.listdir(train_data_location):
                print(f)
                if(f == 'trec_5000_train.txt'):
                    # Subjective Data
                    with open(train_data_location + f, encoding = "ISO-8859-1") as subj_file:
                        for line in subj_file:
                            split_line = line.split(':')#
                            ques_class = split_line[0]
                            question = line.split(' ',1)[1]#split_line[1]
                            pattern = "[^a-zA-Z.' ]"
                            cleaned_line = re.sub(pattern,' ',question)
                            cleaned_line = cleaned_line.lower()
                            x_train.append(cleaned_line)
                            if(ques_class == 'NUM'):
                                y_train.append(0)
                            elif(ques_class == 'DESC'):
                                y_train.append(1)
                            elif(ques_class == 'ENTY'):
                                y_train.append(2)
                            elif(ques_class == 'HUM'):
                                y_train.append(3)
                            elif(ques_class == 'ABBR'):
                                y_train.append(4)
                            elif(ques_class == 'LOC'):
                                y_train.append(5)
                elif(f == 'trec_test.txt'):
                    # Objective Data
                    with open(train_data_location + f, encoding = "ISO-8859-1") as obj_file:
                        for line in obj_file:
                            split_line = line.split(':')#line.split(' ',1)
                            ques_class = split_line[0]
                            question = line.split(' ',1)[1]#split_line[1]
                            pattern = "[^a-zA-Z.' ]"
                            cleaned_line = re.sub(pattern,' ',question)
                            cleaned_line = cleaned_line.lower()
                            x_test.append(cleaned_line)
                            if(ques_class == 'NUM'):
                                y_test.append(0)
                            elif(ques_class == 'DESC'):
                                y_test.append(1)
                            elif(ques_class == 'ENTY'):
                                y_test.append(2)
                            elif(ques_class == 'HUM'):
                                y_test.append(3)
                            elif(ques_class == 'ABBR'):
                                y_test.append(4)
                            elif(ques_class == 'LOC'):
                                y_test.append(5)
            return x_train, y_train, x_test, y_test
          
    def open_pretrained(self):
        """Getting GloVe Embeddings to be used for embedding
        layer. Corresponding words to be feature hashed for look
        up.
        Returns
            NumPy Tensor of shape (300,)
        """
        embeddings = []
        glove_w2emb = {}
        glove_embeddings_file = open(os.path.join('/content/drive/My Drive/College/Undergraduate Research/SkillEvaluation/','glove.6B.'+str(self.cfg['embedding_dim'])+'d.txt'))
        
        # -- Padding --
        glove_w2emb['_pad'] = (0, None)
        
        # -- OOV Words --
        unk_words = np.random.rand(self.cfg['embedding_dim'],)
        glove_w2emb['_unk'] = (1, unk_words)
        embeddings.append(unk_words)
        
        idx = 2
        for line in tqdm(glove_embeddings_file):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_w2emb[word] = (idx , coefs)
            embeddings.append(coefs)
            idx+=1
        glove_embeddings_file.close()
        return glove_w2emb, embeddings
    
    def open_bpe_vectors(self):
        en_model = BPEmb(lang='en',vs=200000,dim=100)
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
            x_train = torch.tensor(x_train).cuda()
            y_train = torch.tensor(y_train,dtype=torch.long).cuda()
            x_test = torch.tensor(x_test).cuda()
            y_test = torch.tensor(y_test,dtype=torch.long).cuda()
        else:
            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train,dtype=torch.long)
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test,dtype=torch.long)

        train_dataloader = self.create_dataloader(features=x_train, labels=y_train)
        test_dataloader = self.create_dataloader(features=x_test, labels=y_test)
        return train_dataloader,test_dataloader

    
    def build_model(self, embeddings):
        return ShallowCNN(embeddings)
    
    def train(self,train_data):

        epochs = 100 # self.cfg['epochs']
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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='min', 
                                                               factor=0.1, 
                                                               patience=5, 
                                                               verbose=True, 
                                                               threshold=0.0001, 
                                                               threshold_mode='rel', 
                                                               cooldown=0, 
                                                               min_lr=0, 
                                                               eps=1e-08)
        #loss_function = torch.nn.CrossEntropyLoss()
        loss_function = torch.nn.NLLLoss()
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            loss = 0
            for i , (examples, labels) in enumerate(train_data):
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
                loss = loss_function(predictions,labels_idx.long())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                #break
            scheduler.step(total_loss) 
            losses.append(total_loss)
            #break
            print('Epoch {} ----> loss={}'.format(epoch,total_loss))
            #print('Epoch {} Learning_Rate{} ----> loss={}'.format(epoch,scheduler.get_lr(),total_loss))
            print('==========================================================')
        return self.model, loss_function, losses
    
    def test_validate(self,debug=False,model=None,test_data=[],loss_fn=None):
        test_loss = 0
        correct = 0
        all_predictions = []
        for idx,(examples, labels) in enumerate(test_data):

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
            # print('correct = ',correct)
            #accuracy = correct.float()/64 * 100

            if(debug):
                for ex,label,label_idx,pred,pred_idx in zip(examples,labels,labels_idx,outputs,preds):
                    print('{}: actual = {} ---> pred = {}'.format(idx,label_idx.item(),pred_idx.item()))
       
        accuracy = correct.float()/500 * 100
        return test_loss, accuracy, all_predictions

def main():
    trainer = Trainer()
    model,criterion,losses = trainer.train(train_data=trainer.train_dataloader)
    test_loss,acc,preds = trainer.test_validate(debug=True,model=trainer.model,test_data=trainer.test_dataloader,loss_fn=criterion)
    print('Test Accuracy = {}%'.format(acc))

if __name__ == '__main__':
    main()