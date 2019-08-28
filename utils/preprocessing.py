import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
class DataPrepper():
    def __init__(self,config={},dataset=None):
        self.config = config
        self.dataset_type = dataset

        if(self.dataset_type == 'trec'):
            self.x_train, self.y_train, self.x_test, self.y_test = self.read_trec_dataset(
                train_data_location=self.config['train_data_location']+self.config['dataset']+'/',
                use_default_split=True
            )
        elif(self.dataset_type == 'subj'):
            dataset,labels = self.read_subj_dataset(train_data_location=self.config['train_data_location']+self.config['dataset']+'/')
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                dataset,
                labels, 
                test_size=0.2, 
                random_state=1000
            )
        pass
    
    def read_subj_dataset(self, train_data_location):
        """Open and prepare the subjectivity dataset. Using
        Regular expressions to clean the sentences.
        
        Args:
            train_data_location - location of the data, specified in config
        Return:
            dataset - ndarray of each example
            labels - array of binary labels
        """
        dataset = []
        labels = []
        for f in os.listdir(train_data_location):
            print(f)
            if(f == 'quote.tok.gt9.5000'):
                # Subjective Data
                with open(train_data_location + f, encoding = "ISO-8859-1") as subj_file:
                    for line in subj_file:
                        pattern = "[^a-zA-Z.' ]"
                        cleaned_line = re.sub(pattern,' ',line)
                        dataset.append(cleaned_line)
                        labels.append(0)
            elif(f == 'plot.tok.gt9.5000'):
                # Objective Data
                with open(train_data_location + f, encoding = "ISO-8859-1") as obj_file:
                    for line in obj_file:
                        pattern = "[^a-zA-Z.' ]"
                        cleaned_line = re.sub(pattern,' ',line)
                        dataset.append(cleaned_line)
                        labels.append(1)
        return np.array(dataset), np.array(labels)
    
    def read_trec_dataset(self, train_data_location, use_default_split=False):
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
            return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)