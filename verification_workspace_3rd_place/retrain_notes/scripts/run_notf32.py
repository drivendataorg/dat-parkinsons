"""Run a repo script with TF32 disabled (cuDNN conv + matmul), without editing the repo.
usage: python run_notf32.py <script.py> [args...]"""
import runpy, sys, torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
