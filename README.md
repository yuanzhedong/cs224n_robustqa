# CS224N default final project (2022 RobustQA track)

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`




## Train baseline

```
python train.py --do-train --run-name baseline_distilbert --model-type distilbert
```


## Train MoE


```
python train.py --do-train --run-name baseline_moe --model-type moe
```

## Tracking experiments with WandB
### Set up the environment
From the command line, install and log in to wandb

```
pip install wandb
wandb login
```

Find your personal API key [](here). Copy this key and paste it into your command line when asked to authorize your account. 

### View the jobs
You can view the jobs in https://wandb.ai/cs224n-robustqa.

### To enable tracking via distributed training
TODO: https://docs.wandb.ai/guides/track/advanced/distributed-training
