# %%
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

from seaborn import lineplot



def read_log(fname: str) -> pd.DataFrame:
    # прочитать логи в пандас
    val_accs = []
    with open(fname) as f:
        for line in f:
            if match := re.match(r"INFO:nas.trainer:eval.*(\[.*\]), (0\.\d*)", line):
                # sampled_gnn = match.group(1)
                val_acc = match.group(2)
                val_accs.append(float(val_acc))
    return pd.DataFrame(val_accs, columns=['val_acc'])


def plot_of_count(vals: pd.DataFrame, min_val: float = 0.8, title: str = None):
    x = vals.index
    y = []
    cnt = 0
    for val in vals.val_acc:
        if min_val <= val:
            cnt += 1
        y.append(cnt)
    df = pd.DataFrame({"step": x, "count": y})
    lineplot(df)

# %%

datasets = ["cora", "bzr", "pubmed", "citeseer", "mutag", "cox2"]
logs_dir = "/home/ubuntu/GNN-AID/src/nas/logs/2_cora_bzr"
logs = {d:[] for d in datasets}
for dir_name, _, fnames in os.walk(logs_dir):
    for f in fnames:
        for d in datasets:
            if d in f:
                logs[d].append(dir_name + f)
                break
    break

# %%
# вывести картинку
min_val = 0.88
df = pd.DataFrame()
for log in logs[""]:
    vals = read_log(log)
    x = vals.index
    if df.empty:
        df = pd.DataFrame(index=x)
    y = []
    cnt = 0
    for val in vals.val_acc:
        if min_val <= val:
            cnt += 1
        y.append(cnt)
    name = log.split('.')[0].split('/')[-1]
    df['count ' + name] = y
lineplot(df).set_title("threshold " + str(min_val))

# %%
