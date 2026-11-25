"""Previously used global-history CNN student, adapted to TimeDistill API.

Each channel shares weights. The number of input channels is dynamic.
The configurable feature is parameter-free interpolation, used only for KD.
"""
import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
 def __init__(self,configs):
  super().__init__()
  if configs.seq_len!=720 or configs.pred_len!=720:
   raise ValueError('CNNGlobal currently supports exactly 720 -> 720')
  width=34
  self.feature_width=configs.d_model
  self.embed=nn.Linear(12,width)
  self.core=nn.Sequential(nn.Conv1d(width,width,5,padding=2),nn.GELU(),nn.Conv1d(width,width,5,padding=2),nn.GELU())
  self.readout=nn.Linear(60*width,720)
 def forward(self,x,*args,**kwargs):
  mean=x.mean(1,keepdim=True).detach()
  a=(x-mean).transpose(1,2)
  batch,channels,_=a.shape
  z=self.embed(a.reshape(batch*channels,60,12))
  z=z+self.core(z.transpose(1,2)).transpose(1,2)
  out=self.readout(z.flatten(1)).reshape(batch,channels,720)
  hidden=z.mean(1).reshape(batch,channels,-1)
  feature=F.interpolate(hidden,size=self.feature_width,mode='linear',align_corners=False)
  return out.transpose(1,2)+mean,[feature,out]
