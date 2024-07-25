workspaceFolder=/home/ubuntu/GNN-AID
PYTHONPATH=${workspaceFolder}:${workspaceFolder}/src
cd $workspaceFolder/src/nas
which python

python -m debugpy --listen 3342 test.py