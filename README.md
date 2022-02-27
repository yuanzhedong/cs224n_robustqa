# CS224N default final project (2022 RobustQA track)

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01`




## Train baseline

```
python train.py --do-train --run-name baseline_distilbert --model-type distilbert
```


## Train MoE


```
python train.py --do-train --run-name baseline_moe --model-type moe
```



## Train switch transformer
```
python train.py --do-train --run-name baseline_switch_transformer --model-type switch_transformer
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

## Train MoE with data augmentation
```
pip install -U nltk
python train.py --do-train --run-name baseline_moe --model-type moe --eda true
```
Given a sentence in the training set, we perform the following operations:

- **Synonym Replacement (SR):** Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
- **Random Insertion (RI):** Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this *n* times.
- **Random Swap (RS):** Randomly choose two words in the sentence and swap their positions. Do this *n* times.
- **Random Deletion (RD):** For each word in the sentence, randomly remove it with probability *p*.

There are 5 parameters that can be adjusted:
`num_aug, alpha_sr, alpha_ri, alpha_rs, alpha_rd`

So for example, if you want `16` augmented sentences per original sentence and replace 5% of words by synonyms (`alpha_sr=0.05`), delete 10% of words (`alpha_rd=0.1`) and do not apply random insertion (`alpha_ri=0.0`) and random swap (`alpha_rs=0.0`), you would do:
`--num_aug=16 --alpha_sr=0.05 --alpha_rd=0.1 --alpha_ri=0.0 --alpha_rs=0.0`
