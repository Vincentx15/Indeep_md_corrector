install conda then
```shell
conda create -n md_ranker -y
conda activate md_ranker
conda install pymol pymol-psico -c schrodinger
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
pip install mrcfile
```