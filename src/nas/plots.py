# %%
import pandas as pd
import re

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

logs_dir = "/home/ubuntu/GNN-AID/src/nas/logs/"
cora_logs = [
    logs_dir + "cora_basic.log",
    logs_dir + "cora_fixed.log",
    logs_dir + "cora_dynamic.log",
]

bzr_logs = [
    logs_dir + "bzr_basic.log",
    logs_dir + "bzr_fixed.log"
]
# %%
# вывести картинку
min_val = 0.87
df = pd.DataFrame()
for log in bzr_logs:
    vals = read_log(log)
    x = vals.index
    if df.empty:
        df = pd.DataFrame(index=x[:500])
    y = []
    cnt = 0
    for val in vals.val_acc:
        if min_val <= val:
            cnt += 1
        y.append(cnt)
    name = log.split('.')[0].split('/')[-1]
    df['count ' + name] = y[:500]
lineplot(df)
