"""Offline preparation checks; never trains or evaluates a model."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parent
class PreparationChecks(unittest.TestCase):
 def test_five_dataset_dry_runs(self):
  for name in ('ETTm2','ECL','Solar','Traffic','Weather'):
   with self.subTest(dataset=name):
    result=subprocess.run([sys.executable,str(ROOT/'prepare_five.py'),'--dataset',name],cwd=ROOT,text=True,capture_output=True)
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertIn('DRY RUN: no training or testing started',result.stdout)
    self.assertIn('teacher_matches: 0',result.stdout)
 def test_teacher_script_paths(self):
  for name in ('ETTm2','ECL','Solar','Traffic','Weather'):
   self.assertTrue((ROOT/'scripts'/'long_term_forecast'/name/'ModernTCN.sh').is_file())
  text=(ROOT/'scripts'/'long_term_forecast'/'Solar'/'ModernTCN.sh').read_text()
  self.assertIn('./dataset/solar/',text)
 def test_train_loop_does_not_construct_or_evaluate_test(self):
  p=ROOT/'exp'/'exp_long_term_forecasting.py';tree=ast.parse(p.read_text(encoding='utf-8'))
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Exp_Long_Term_Forecast')
  fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
  text=ast.get_source_segment(p.read_text(encoding='utf-8'),fn)
  self.assertNotIn("flag='test'",text)
  self.assertNotIn('test_loader',text)
  self.assertNotIn('test_data',text)
 def test_missing_teacher_blocks_execution(self):
  result=subprocess.run([sys.executable,str(ROOT/'prepare_five.py'),'--dataset','ETTm2','--execute'],cwd=ROOT,text=True,capture_output=True)
  self.assertNotEqual(result.returncode,0)
  self.assertIn('Exactly one matching teacher checkpoint is required',result.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
