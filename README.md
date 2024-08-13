
### Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name teast_env python=3.8
source activate teast_env
conda install --file requirements.txt -c pytorch
```


### Datasets

```
python process_icews.py

python process_gdelt.py
```

This will create the files required to compute the filtered metrics.

### Reproducing results of TCompoundE

In order to reproduce the results of TCompoundE on the four datasets in the paper,  run the following commands

```
python  learner.py --dataset ICEWS14 --emb_reg 0.01 --time_reg 0.01 --learning_rate 0.01  --rank 6000  --batch_size 4000  --max_epochs 400

python  learner.py --dataset ICEWS05-15 --emb_reg 0.002   --time_reg 0.1  --learning_rate 0.08  --rank 8000  --batch_size 6000  --max_epochs 100

python  learner.py --dataset GDELT --emb_reg 0.001   --time_reg 0.001  --learning_rate 0.35  --rank 6000  --batch_size 2000  --max_epochs 50
```

