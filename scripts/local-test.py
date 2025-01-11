import sys
from pathlib import Path

sys.path.append("src/discord-cluster-manager")

from leaderboard_eval import cu_eval
from run_eval import run_cuda_script

ref = Path("examples/identity_cuda/reference.cuh")
sub = Path("examples/identity_cuda/submission.cuh")

cout, score = run_cuda_script(cu_eval, ref.read_text(), sub.read_text(), arch=80)
print(cout)
print(score)
