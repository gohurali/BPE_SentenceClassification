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

class Inferencer():

    def __init__(self):
        pass

    def inference(self):
        pass

def main():
    pass

if __name__ == '__main__':
    main()