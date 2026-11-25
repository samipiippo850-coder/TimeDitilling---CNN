"""Synthetic CNN+ModernTCN GPU compatibility/memory preflight (no real data).

Runs one optimizer step on random models only; no checkpoint/test is read.
Results must not be interpreted as training quality or as a concurrency bound.
"""
import argparse, importlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'audit_reference'))
from models.CNNGlobal import Model as CNN
from models.ModernTCN import Model as Teacher
from models.Regressor import Model as Regressor
from multiarch_global_students_20261031 import Student as ReferenceCNN
from exp.exp_long_term_forecasting import multi_scale_process_inputs

DATA={'ETTm2':(7,512),'ECL':(321,512),'Solar':(137,512),'Traffic':(862,1024),'Weather':(21,512)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATA,required=True);p.add_argument('--batch',type=int,default=1)
 a=p.parse_args();assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(2021)
 channels,dim=DATA[a.dataset];conf=json.loads((ROOT/'config.json').read_text())[a.dataset]['ModernTCN']
 base=dict(stem_ratio=6,downsample_ratio=2,dw_dims=[256]*4,enc_in=channels,small_kernel_merged=False,
  dropout=.3,use_multi_scale=False,revin=1,affine=0,subtract_last=0,freq='h',seq_len=720,
  individual=0,pred_len=720,kernel_size=25,decomposition=0)
 base.update(conf);tc=SimpleNamespace(**base);sc=SimpleNamespace(seq_len=720,pred_len=720,d_model=dim)
 device=torch.device('cuda');torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 student=CNN(sc).to(device);ref=ReferenceCNN('cnn',34,feature_width=dim).to(device);ref.load_state_dict(student.state_dict(),strict=True)
 student.eval();ref.eval();small=torch.randn(1,720,min(channels,7),device=device)
 with torch.no_grad():
  out,sf=student(small);refout,reff=ref(small)
  assert torch.equal(out,refout) and torch.equal(sf[0],reff[0])
 del ref,small,out,sf,refout,reff
 teacher=Teacher(tc).to(device);teacher.train();x=torch.randn(a.batch,720,channels,device=device);y=torch.randn_like(x)
 opt_t=torch.optim.Adam(teacher.parameters(),lr=1e-4);tp,_=teacher(x);tl=F.mse_loss(tp,y);tl.backward();opt_t.step();torch.cuda.synchronize()
 teacher_peak=torch.cuda.max_memory_allocated();del opt_t,tp,tl
 teacher.eval();teacher.requires_grad_(False);reg=Regressor(sc,tc.dims[-1]*(720//tc.patch_stride),dim).to(device)
 opt=torch.optim.Adam(list(student.parameters())+list(reg.parameters()),lr=.01);student.train()
 with torch.no_grad():tp,tf=teacher(x)
 pred,sf=student(x);mapped=reg(tf[-2].flatten(-2));assert pred.shape==y.shape and sf[-2].shape==mapped.shape
 down=multi_scale_process_inputs();s=down(pred.transpose(1,2));t=down(tp.transpose(1,2));out_time=sum(F.mse_loss(q,w) for q,w in zip(s,t))/len(s)
 s=down(sf[-2]);t=down(mapped);feat_time=sum(F.mse_loss(q,w) for q,w in zip(s,t))/len(s)
 pa=torch.fft.rfft(pred,dim=1).abs()[:,1:];ta=torch.fft.rfft(tp,dim=1).abs()[:,1:]
 out_freq=F.kl_div(torch.log(F.softmax(pa/.5,dim=1)+1e-8),F.softmax(ta/.5,dim=1),reduction='mean')
 pa=torch.fft.rfft(sf[-2].transpose(1,2),dim=1).abs()[:,1:];ta=torch.fft.rfft(mapped.transpose(1,2),dim=1).abs()[:,1:]
 feat_freq=F.kl_div(F.log_softmax(pa/.5,dim=1),F.log_softmax(ta/.5,dim=1),reduction='mean',log_target=True)
 loss=F.mse_loss(pred,y)+out_time+out_freq+feat_time+feat_freq
 assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
 opt.step();torch.cuda.synchronize()
 report={'dataset':a.dataset,'batch':a.batch,'device':torch.cuda.get_device_name(0),'student_parameters':sum(p.numel() for p in student.parameters()),
  'feature_shape':list(mapped.shape),'teacher_train_peak_bytes':teacher_peak,'total_peak_bytes':torch.cuda.max_memory_allocated(),
  'seconds':time.monotonic()-start,'reference_cnn_exact':True,'teacher_random':True,'data_read':False,'finite_loss':True}
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
