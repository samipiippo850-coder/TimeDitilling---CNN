"""Inspect the five prepared datasets and print a train-only command.

Default is read-only/dry-run. --execute starts one explicitly selected run.
No test evaluation is performed by this launcher. The student is CNNGlobal,
adapted from the earlier global-history CNN and independently preflighted.
"""
import argparse
import glob
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DATA={
 'ETTm2':('ETT-small','ETTm2.csv','ETTm2',7,512,'non-stationary',2,.5,20,32,'db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd'),
 'ECL':('electricity','electricity.csv','custom',321,512,'non-stationary',.1,.5,20,32,'7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad'),
 'Solar':('solar','solar_AL.txt','Solar',137,512,'non-stationary',.1,2,20,32,'230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2'),
 'Traffic':('traffic','traffic.csv','custom',862,1024,'revin',.1,.1,10,8,'cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16'),
 'Weather':('weather','weather.csv','custom',21,512,'non-stationary',.5,2,20,32,'34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63'),
}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',choices=DATA,required=True)
 p.add_argument('--mode',choices=('teacher','distill','supervised'),default='distill')
 p.add_argument('--horizon',type=int,choices=(96,192,336,720),default=720)
 p.add_argument('--epochs',type=int,default=None)
 p.add_argument('--batch-size',type=int,default=None,help='override per-dataset batch; applies to teacher and student')
 p.add_argument('--gpu',type=int,default=0)
 p.add_argument('--execute',action='store_true',help='explicitly start a single training run')
 a=p.parse_args(); folder,filename,data,n,d,norm,alpha,beta,epochs,batch,expected=DATA[a.dataset]
 path=ROOT/'dataset'/folder/filename
 if not path.is_file():raise SystemExit('Missing dataset: '+str(path))
 got=sha(path)
 if got!=expected:raise SystemExit('Dataset hash mismatch: '+str(path))
 epochs=a.epochs if a.epochs is not None else epochs
 if epochs<1:raise SystemExit('epochs must be positive')
 batch=a.batch_size if a.batch_size is not None else batch
 if batch<1:raise SystemExit('batch-size must be positive')
 teacher_glob=str(ROOT/'checkpoints'/f'long_term_forecast_{a.dataset}_720_{a.horizon}_ModernTCN_{data}*'/'checkpoint.pth')
 teachers=sorted(glob.glob(teacher_glob)) if a.mode in ('teacher','distill') else []
 cmd=[sys.executable,'-u','run.py','--task_name','long_term_forecast','--is_training','1',
      '--root_path',str(path.parent),'--data_path',filename,'--model_id',f'{a.dataset}_720_{a.horizon}',
      '--model','CNNGlobal','--model_t','ModernTCN','--data',data,'--features','M','--seq_len','720',
      '--label_len','48','--pred_len',str(a.horizon),'--e_layers','2','--factor','3',
      '--enc_in',str(n),'--dec_in',str(n),'--c_out',str(n),'--des','Exp','--itr','1',
      '--learning_rate','0.01','--lradj','type1','--d_model',str(d),'--train_epochs',str(epochs),
      '--norm',norm,'--alpha',str(alpha),'--beta',str(beta),'--batch_size',str(batch),'--num_workers','0']
 if a.mode=='distill':cmd.append('--distillation')
 if a.mode=='teacher':
  if a.epochs is not None:raise SystemExit('Teacher epoch count is defined by the upstream per-dataset script; --epochs is student-only.')
  cmd=['bash',str(ROOT/'scripts'/'long_term_forecast'/a.dataset/'ModernTCN.sh'),str(a.gpu),str(a.horizon),'720',str(batch)]
 print('dataset_sha256:',got,'bytes:',path.stat().st_size)
 print('mode:',a.mode,'student: CNNGlobal width 34','teacher_matches:',len(teachers))
 print('teacher_glob:',teacher_glob if a.mode in ('teacher','distill') else '(not required)')
 print('command:',subprocess.list2cmdline(cmd))
 print('note: training evaluates validation only; test_after_train is not enabled; CSV loader may read entire source file.')
 if not a.execute:
  print('DRY RUN: no training or testing started');return
 if a.mode=='distill' and len(teachers)!=1:raise SystemExit('Exactly one matching teacher checkpoint is required before training.')
 if a.mode=='teacher' and teachers:raise SystemExit('Matching teacher checkpoint already exists; refusing to overwrite.')
 if a.mode=='teacher' and not shutil.which('bash'):raise SystemExit('bash is required for the original teacher training script.')
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 try:import torch
 except ImportError as e:raise SystemExit('PyTorch is unavailable; install the project environment before executing.') from e
 if not torch.cuda.is_available():raise SystemExit('CUDA is unavailable; refusing unintended CPU training.')
 env=os.environ.copy()
 subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
