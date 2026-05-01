# %%
import re
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# %%
class MultiHeadedAttention(LightningModule):
    def __init__(self, d_in,d_out,num_of_heads):
        super().__init__()
        self.d_out_total=d_out*num_of_heads
        self.num_of_heads=num_of_heads
        self.d_out=d_out
        self.dropout_layer=torch.nn.Dropout(0.1)
        self.W_query=nn.Linear(d_in,self.d_out_total)
        self.W_key=nn.Linear(d_in,self.d_out_total)
        self.W_value=nn.Linear(d_in,self.d_out_total)
        self.projection_layer=nn.Linear(self.d_out_total,self.d_out)

    def forward(self,x):
        B,T,_=x.size()
        H,Dh=self.num_of_heads,self.d_out
        queries=self.W_query(x).view(B,T,H,Dh).transpose(1,2)
        keys=self.W_key(x).view(B,T,H,Dh).transpose(1,2)
        values=self.W_value(x).view(B,T,H,Dh).transpose(1,2)

        #Equivalent to .transpose(-1,-2) since it is symmetric
        attn_scores=queries@(keys.transpose(2,3))
        #print("attn_scores: "+str(attn_scores.size()))
        #attn_scores.size=B,H,T,T

        mask_length=attn_scores.size()[-1]
        mask_simple=torch.triu(torch.ones(mask_length,mask_length),diagonal=1).to(x.device)
        mask_simple=mask_simple.masked_fill(mask_simple.bool(),-torch.inf)

        masked_attn_scores=(mask_simple+attn_scores)/((keys.size()[-1])**0.5)
        attn_weights=torch.softmax(masked_attn_scores,dim=-1)
        masked_dropout_attn_weights=self.dropout_layer(attn_weights)

        context_vec=(masked_dropout_attn_weights@values).transpose(1,2)
        context_vec=context_vec.contiguous().view(B, T, self.d_out_total)
        context_vec=self.projection_layer(context_vec)
        #print("context_vec_size:" +str(context_vec.size()))
        #context_vec.size=B,T,d_out

        return context_vec
    
class TransformerBlock(LightningModule):
    def __init__(self,d_in,d_out,number_of_heads):
        super().__init__()
        self.MultiHeadedAttention=MultiHeadedAttention(d_in,d_out,number_of_heads)
        self.GELU=torch.nn.GELU()
        self.LinearBlock=nn.Linear(d_out,d_out)

    def batch_normalization(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True)
        norm_data=(x-mean)/torch.sqrt(var)
        return norm_data
    

    def forward(self,x):
        x1=self.batch_normalization(x)
        x1=self.MultiHeadedAttention(x1).to(x.device)
        x1=x1+x
        x2=self.batch_normalization(x1)
        x2=self.LinearBlock(x2)
        x2=self.GELU(x2)
        x2=self.LinearBlock(x2)
        x2=x2+x1
        return x2


class GhostGPT(LightningModule):
    def __init__(self,d_in,d_out,num_blocks,number_of_heads=12,embedding_dim=5,flattened_image_size=106*106,context_size=154,final_image_size=256*256):
        super().__init__()
        self.main_body=nn.ModuleList([TransformerBlock(d_in,d_out,number_of_heads) for i in range(num_blocks)])
        self.call_transformer=TransformerBlock(d_in,d_out,number_of_heads)
        self.final_projection_layer=nn.Linear(d_out,16)
        self.final_projection_layer2=nn.Linear(context_size*16,final_image_size)
        self.final_sigmoid_layer=nn.Sigmoid()
        self.image_embedding_layer=torch.nn.Linear(flattened_image_size,embedding_dim-1)
        self.pos_embedding_layer=torch.nn.Embedding(context_size,embedding_dim)
        self.context_size=context_size
        self.embedding_dim=embedding_dim

    def forward(self,x,bucket_sum):
        x=self.image_embedding_layer(x).expand(bucket_sum.size()[0],self.context_size,self.embedding_dim-1)
        bucket_sum=bucket_sum.view(bucket_sum.size()[0],self.context_size,1)
        x=torch.cat([x,bucket_sum],dim=-1)
        x=x+self.pos_embedding_layer(torch.arange(x.size()[1],device=x.device))
        for modules in self.main_body:
            x=modules(x)
        x=self.call_transformer.batch_normalization(x)
        x=self.final_projection_layer(x)
        x=x.view(x.size()[0],-1)
        x=self.final_projection_layer2(x)
        x=self.final_sigmoid_layer(x)
        return x
