import torch
import torch.nn.functional as F
import yaml

class ShallowCNN(torch.nn.Module):
    def __init__(self,config={},pretrained_embeddings=None):
        super(ShallowCNN,self).__init__()
        #self.cfg = yaml.safe_load(open('config.yaml'))
        self.cfg = config
        self.num_classes = self.cfg['num_classes']
        
        # -- Build Embedding Table --
        self.pretrained_embedding_table = torch.nn.Embedding.from_pretrained(
            embeddings=pretrained_embeddings,
            freeze=True,
            padding_idx=None, 
            max_norm=None, 
            norm_type=2.0, 
            scale_grad_by_freq=False, 
            sparse=False
            )
        
        # -- Define Architecture --
        self.conv1 = torch.nn.Conv1d(in_channels=self.cfg['pad_limit'],
                                     out_channels=400,
                                     kernel_size=(4,),
                                     stride=1,
                                     padding=0,
                                     bias=True
                                    )
        self.mp1 = torch.nn.MaxPool1d(kernel_size=2,
                                      stride=1,
                                      padding=0
                                     )
        self.fc1 = torch.nn.Linear(in_features=38400,#118400,#self.cfg['embedding_dim'] - 4,
                                       out_features=128, 
                                       bias=True
                                      )
        if(self.cfg['if_softmax']):
            self.fc2 = torch.nn.Linear(in_features=128,
                                       out_features=6,
                                       bias=True
                                      )
        else:
            self.fc2 = torch.nn.Linear(in_features=128,
                                       out_features=1,
                                       bias=True
                                      )
    def forward(self, inputs):
        """Forward pass definition
        Args:
            inputs - Array of indices for embeddings lookup
        """
        emb = self.pretrained_embedding_table(inputs)
        x = F.leaky_relu(self.conv1(emb))
        x = self.mp1(x)
        x = x.view(x.shape[0],-1)
        #print('Flatten = ', x.shape)
        
        x = F.leaky_relu(self.fc1(x))
        if(self.cfg['if_softmax']):
            x = self.fc2(x)
            x = F.log_softmax(x,dim=1,dtype=torch.float)
        else:
            x = self.fc2(x)
            x = torch.sigmoid(x)
        return x
