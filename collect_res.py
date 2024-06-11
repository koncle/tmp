import argparse
import os
from pathlib import Path
import re

import numpy as np


lines = """
first grad norm: 1.3294109106063843, second grad norm: 1.673986792564392, cos_sim: 0.8913594484329224┃         189  ┃      0.0808  │     97.11 %  ┃   6.156e-04  │   01:15 min  ┃      0.1780  │     92.08 %  ┃
first grad norm: 1.2486217021942139, second grad norm: 1.7005512714385986, cos_sim: 0.824838399887085┃         190  ┃      0.0804  │     97.15 %  ┃   4.988e-04  │   01:15 min  ┃      0.1780  │     91.93 %  ┃
first grad norm: 1.6147626638412476, second grad norm: 1.8176627159118652, cos_sim: 0.872918963432312┃         191  ┃      0.0789  │     97.27 %  ┃   3.943e-04  │   01:15 min  ┃      0.1771  │     92.03 %  ┃
first grad norm: 1.1026198863983154, second grad norm: 1.3242353200912476, cos_sim: 0.8517789840698242┃         192  ┃      0.0774  │     97.25 %  ┃   3.020e-04  │   01:06 min  ┃      0.1767  │     92.04 %  ┃
first grad norm: 1.1739836931228638, second grad norm: 1.7602969408035278, cos_sim: 0.7693657875061035┃         193  ┃      0.0792  │     97.18 %  ┃   2.219e-04  │   01:02 min  ┃      0.1760  │     92.14 %  ┃
first grad norm: 1.3375625610351562, second grad norm: 1.9825495481491089, cos_sim: 0.7593687772750854┃         194  ┃      0.0770  │     97.29 %  ┃   1.541e-04  │   01:10 min  ┃      0.1771  │     92.17 %  ┃
first grad norm: 1.242916464805603, second grad norm: 1.690833330154419, cos_sim: 0.9032294750213623┃         195  ┃      0.0794  │     97.11 %  ┃   9.866e-05  │   01:03 min  ┃      0.1779  │     92.02 %  ┃
first grad norm: 1.3216979503631592, second grad norm: 1.505342960357666, cos_sim: 0.8790898323059082┃         196  ┃      0.0790  │     97.12 %  ┃   5.551e-05  │   01:15 min  ┃      0.1768  │     92.24 %  ┃
first grad norm: 1.4286350011825562, second grad norm: 1.9302735328674316, cos_sim: 0.8808057904243469┃         197  ┃      0.0762  │     97.34 %  ┃   2.467e-05  │   01:14 min  ┃      0.1763  │     92.01 %  ┃
first grad norm: 0.908609926700592, second grad norm: 1.7449150085449219, cos_sim: 0.717512309551239┃         198  ┃      0.0790  │     97.16 %  ┃   6.168e-06  │   01:14 min  ┃      0.1770  │     92.09 %  ┃
first grad norm: 1.0207469463348389, second grad norm: 1.5319768190383911, cos_sim: 0.8775969743728638┃         199  ┃      0.0770  │     97.26 %  ┃   0.000e+00  │   01:09 min  ┃      0.1772  │     92.10 %  ┃
"""


# extract the last accuracy with re

def extract_acc(path):
    
    # find latest .txt file with getmtime
    latest_file = None
    for f in Path(path).iterdir():
        if f.is_file() and f.suffix == '.txt':
            if not latest_file:
                latest_file = f
            else:
                if os.path.getmtime(f) > os.path.getmtime(latest_file): 
                    latest_file = f
    # print(latest_file)
    accs = []
    with open(latest_file, 'r') as f:
        lines = f.readlines()
        for line in lines:            
            find_res = re.findall(r'(\d+\.\d+)', line)
            if len(find_res) > 3:                
                acc = float(find_res[-1])
                accs.append(acc)
    # print(accs)
    return np.max(accs)


def extract_all(root, filter=''):
    for path in Path(root).iterdir():
        if len(filter) > 0 and filter not in path.name:
            continue
        accs = []
        for i in range(1, 4):
            p = path / str(i)
            print(p) 
            acc = extract_acc(p)
            accs.append(acc)
        mean = np.mean(accs)
        std = np.std(accs)
        print(accs) 
        print(f'{path.name}: ${{{mean:.2f}}}_{{\pm{std:.2f}}}$ \n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default='', type=str)
    args = parser.parse_args()
    
    extract_all('outputs/cifar10', filter=args.f)
    print('\n\n')
    extract_all('outputs/cifar100', filter=args.f)
