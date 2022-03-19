# Build a Robust QA System with Transformer-based Mixture of Experts

## Dataset
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)

## Setup
### Conda
- Setup environment with `conda env create -f environment.yml`.
### Docker
- Build the docker with the `Dockerfile`.

## Train baseline

```
python train.py --do-train --run-name baseline_distilbert --model-type distilbert
```


## Train MoE

Without pretraining:
- All of the indomain_train and oodomain_train datasets will be used for training
```
# without data aug
python train.py --do-train --run-name baseline_moe --model-type moe
# with data aug
python train.py --do-train --run-name baseline_moe --model-type moe --eda --num_aug 4
```

With pretraining:
- The indomain_train and oodomain_train combined will be used for pretraining
- The oodomain_train datasets will be used for finetuning
- The base model and the experts weights will be frozen after pretraining

```
# without data aug
python train.py --do-train --run-name baseline_moe --model-type moe --pretrain True --freeze_expert
python train.py --do-train --run-name pretrain_moe --model-type moe --pretrain --num_experts 1 --num-epochs-pretrain 5 --num-epochs 100 --freeze_expert
# with data aug: 4 for pretraining and 8 for finetuning
python train.py --do-train --run-name pretrain_moe --model-type moe --pretrain --num_experts 1 --num-epochs-pretrain 5 --num-epochs 100 --freeze_expert --eda --num_aug_pretrain 4 --num_aug 8
```

## Train switch transformer
```
python train.py --do-train --run-name baseline_switch_transformer --model-type switch_transformer
# Pretraining
python train.py --do-train --run-name pretrain_moe --model-type moe --pretrain --num_experts 1 --num-epochs-pretrain 5 --num-epochs 100
```

## Train MoE with data augmentation
```
python train.py --do-train --run-name baseline_moe --model-type moe --back_translate --languages es fr
```
For back translation through both Spanish and French. Now back translation only applies to finetuning train set. It takes \~ 20 min to get the default back translated through Spanish train set. Time increases linearly with the number of languages.

Cache can be found in: `cache/finetune/_duorc_race_relation_extraction_back_translate_es_fr_encodings.pt`

```
python train.py --do-train --run-name baseline_moe --model-type moe --eda
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

If you want to use cached data without recomputing the data augmentation, you can add an argument `--use_cache`.

```
python train.py --do-train --run-name baseline_moe --model-type moe --eda --use_cache
```
