import argparse
import json
import os
from collections import OrderedDict
from tkinter import N
import torch
import csv
import util
from pathlib import Path
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from models import MoE
from args import get_train_test_args, DATASET_CONFIG
import torch.nn as nn

from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
# for data augmentation
import perform_eda
import perform_back_translate
import wandb
from switch_transformer import SwitchTransformer, SwitchTransformerLayer, MultiHeadAttention, SwitchFeedForward, FeedForward



def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st: offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(
        f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples


def read_and_process(args, tokenizer, dataset_dict, cache_path, split):
    if split in ['train', 'finetune']:
        tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        print("saving encodings at", cache_path, "...")
        Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
        util.save_pickle(tokenized_examples, cache_path)
    else:
        tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
    return tokenized_examples


# TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.num_epochs_pretrain = args.num_epochs_pretrain
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        
        os.makedirs(self.path, exist_ok=True)
        self.model_type = args.model_type

    def save(self, model, best_scores):
        if self.model_type == "distilbert":
            #model.save_pretrained(self.path)
            f1_score = best_scores["F1"]
            torch.save(model.state_dict(), self.path + f"/model_f1_{f1_score}.pt")
        else:
            print(f"Unsupported model type: {self.model_type}")
        # TODO: add save model
        # save_file = os.path.join(self.save_dir, "saved_model_{:.3f}.pt".format(f1_score))
        # save_file_config = os.path.join(self.save_dir, "config_{:.3f}.json".format(f1_score))
        # model_to_save = model.module if hasattr(model, 'module') else model
        # torch.save(model_to_save.state_dict(), save_file)
        # model_to_save.config.to_json_file(save_file_config)
        return

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    # eval on devset
    def evaluate_moe(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                start_logits, end_logits, loss = model(batch)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                data_loader.dataset.encodings,
                                                (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results
    
    # Save test set results to disk for MoE
    def test_moe(self, model, eval_loader, eval_dict, save_dir):
        eval_preds, eval_scores = self.evaluate_moe(model, eval_loader, eval_dict, return_preds=True, split="test")
        results_str = ', '.join(
            f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        self.log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(save_dir, "test")
        self.log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


    def train(self, model, train_dataloader, eval_dataloader, val_dict, ood_dev_dataloader, ood_dev_dict, rank, world_size):
        #device = self.device
        #model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        global_idx_count = 1
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = None
        if rank == 0:
            tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            if rank == 0:
                self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=int(len(train_dataloader.dataset))) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(rank)
                    attention_mask = batch['attention_mask'].to(rank)
                    start_positions = batch['start_positions'].to(rank)
                    end_positions = batch['end_positions'].to(rank)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    
                    progress_bar.update(len(input_ids) * world_size)
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    if rank == 0:
                        tbx.add_scalar('train/NLL', loss.item(), global_idx)
                        wandb.log({
                        "index": global_idx,
                        "train/NLL": loss.item(),
                        })
                    if (global_idx >= global_idx_count * self.eval_every) and rank == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        global_idx_count += 1
                        preds, curr_score = self.evaluate(
                            model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(
                            f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                            wandb.log({f'val/{k}': v})
                        self.log.info(f'In domain {results_str}')

                        preds, curr_score = self.evaluate(
                            model, ood_dev_dataloader, ood_dev_dict, return_preds=True)
                        results_str = ', '.join(
                            f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'oodomain_val/{k}', v, global_idx)
                            wandb.log({f'oodomain_val/{k}': v})
                        self.log.info(f'Out of domain {results_str}')

                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model, best_scores)
                        for k, v in best_scores.items():
                            wandb.log({f'oodomain_val/best_{k}': v})
                    if rank == 0:
                        global_idx += world_size
        return best_scores

    def train_moe(
        self,
        model,
        pretrain_dataloader,
        train_dataloader,
        dev_dataloader,
        dev_dict,
        ood_dev_dataloader,
        ood_dev_dict,
        test_dataloader,
        test_dict,
        rank,
        world_size,
    ):
        optim = AdamW(model.parameters(), lr=self.lr)
        tbx = None
        if rank == 0:
            tbx = SummaryWriter(self.save_dir)
        best_scores = {'F1': -1.0, 'EM': -1.0}
        if pretrain_dataloader is not None:
            pretrain_step_idx = 0
            pretrain_eval_count = 0
            for epoch_num in range(self.num_epochs_pretrain):
                if rank == 0:
                    self.log.info(f'Pretraing Epoch: {epoch_num}')
                    wandb.log({'Pretraing Epoch': epoch_num})
                with torch.enable_grad(), tqdm(total=len(pretrain_dataloader.dataset)) as progress_bar:
                    for batch in pretrain_dataloader:
                        optim.zero_grad()
                        model.train()
                        input_ids = batch['input_ids'].to(rank)
                        start_logits, end_logits, loss = model(batch)
                        loss.backward()
                        optim.step()
                        if rank == 0:
                            progress_bar.update(len(input_ids)*world_size)
                            progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                            tbx.add_scalar('pretrain/NLL', loss.item(), pretrain_step_idx)
                            wandb.log({
                                "index": pretrain_step_idx,
                                "pretrain/NLL": loss.item(),
                            })
                        # a hard coded eval step size for now
                        if (pretrain_step_idx >= pretrain_eval_count * 10000) and rank == 0:                    
                            self.log.info(f'Evaluating at step {pretrain_step_idx}...')
                            pretrain_eval_count += 1
                            preds, curr_score = self.evaluate_moe(
                                model, dev_dataloader, dev_dict, return_preds=True)
                            results_str = ', '.join(
                                f'{k}: {v:05.2f}' for k, v in curr_score.items())
                            for k, v in curr_score.items():
                                tbx.add_scalar(f'pretrain_val/{k}', v, pretrain_step_idx)
                                wandb.log({f'pretrain_val/{k}': v})
                            self.log.info(f'Pretrain-In domain {results_str}')

                            preds, curr_score = self.evaluate_moe(
                                model, ood_dev_dataloader, ood_dev_dict, return_preds=True)
                            results_str = ', '.join(
                                f'{k}: {v:05.2f}' for k, v in curr_score.items())
                            for k, v in curr_score.items():
                                tbx.add_scalar(f'pretrain_oodomain_val/{k}', v, pretrain_step_idx)
                                wandb.log({f'pretrain_oodomain_val/{k}': v})
                            self.log.info(f'Pretrain-Out of domain {results_str}')

                            if curr_score['F1'] >= best_scores['F1']:
                                best_scores = curr_score
                                self.log.info("Infer on testset...")
                                self.test_moe(model, test_dataloader, test_dict, self.save_dir)
                            for k, v in best_scores.items():
                                wandb.log({f'oodomain_val/pretrain_best_{k}': v})
                        if rank == 0:
                            pretrain_step_idx += world_size
        if args.freeze_basemodel:
            model.freeze_base_model()
        if args.freeze_expert:
            model.freeze_experts()
        global_idx = 0
        global_idx_count = 1
        for epoch_num in range(self.num_epochs):
            if rank == 0:
                self.log.info(f'Epoch: {epoch_num}')
                wandb.log({'Epoch': epoch_num})
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(rank)
                    start_logits, end_logits, loss = model(batch)
                    if loss is None:
                        continue
                    loss.backward()
                    optim.step()
                    if rank == 0:
                        progress_bar.update(len(input_ids)*world_size)
                        progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                        tbx.add_scalar('train/NLL', loss.item(), global_idx)
                        wandb.log({
                            "index": global_idx,
                            "train/NLL": loss.item(),
                        })
                    if (global_idx >= global_idx_count * self.eval_every) and rank == 0:                    
                        self.log.info(f'Evaluating at step {global_idx}...')
                        global_idx_count += 1
                        preds, curr_score = self.evaluate_moe(
                            model, dev_dataloader, dev_dict, return_preds=True)
                        results_str = ', '.join(
                            f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                            wandb.log({f'val/{k}': v})
                        self.log.info(f'In domain {results_str}')

                        preds, curr_score = self.evaluate_moe(
                            model, ood_dev_dataloader, ood_dev_dict, return_preds=True)
                        results_str = ', '.join(
                            f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'oodomain_val/{k}', v, global_idx)
                            wandb.log({f'oodomain_val/{k}': v})
                        self.log.info(f'Out of domain {results_str}')

                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=dev_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            #self.save(model, curr_score['F1'])
                            self.log.info("Infer on testset...")
                            self.test_moe(model, test_dataloader, test_dict, self.save_dir)
                        for k, v in best_scores.items():
                            wandb.log({f'oodomain_val/best_{k}': v})
                    if rank == 0:
                        global_idx += world_size
        return best_scores


def get_dataset(args, tokenizer, split_name, num_aug=0):
    if split_name not in DATASET_CONFIG:
        return
    dataset_paths = DATASET_CONFIG[split_name]
    dataset_dict = None

    datasets_name = ''
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        datasets_name += f'_{dataset_name}'

    if args.eda and num_aug > 0:
        datasets_name += '_eda' + f'_{num_aug}_{args.alpha_sr}_{args.alpha_ri}_{args.alpha_rs}_{args.alpha_rd}'

    if args.back_translate:
        datasets_name += '_back_translate'
        for language in args.languages:
            datasets_name += f'_{language}'

    data_dir = f"cache/{split_name}"
    cache_path = f'{data_dir}/{datasets_name}_encodings.pt'
    print("cache path:", cache_path)

    if split_name in ["train", "finetune"] and os.path.exists(cache_path) and args.use_cache: 
        # avoid recomputing encodings.pt
        print("loading existing", cache_path, "...")
        data_encodings = util.load_pickle(cache_path)

    else:
        print("not using cache, creating new encoding...")
        for dataset_path in dataset_paths:
            dataset_name = os.path.basename(dataset_path)

            # for finetuning, back translate first because it is slower than eda
            if args.back_translate and split_name == "finetune": # also apends orignal sentences
                print("BACK-TRANSLATE", split_name)
                dataset_dict_curr = perform_back_translate.perform_back_translate(
                    args, dataset_path, dataset_name
                )
                dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

            if args.eda and split_name in ["train", "finetune"]:  # eda.py does appends original sentences to augmented sentences
                dataset_dict_curr = perform_eda.perform_eda(
                    args, dataset_path, dataset_name
                )
                dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

            if split_name != "train" or (not args.eda and not args.back_translate):
                dataset_dict_curr = util.read_squad(dataset_path)
                dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

        data_encodings = read_and_process(
            args, tokenizer, dataset_dict, cache_path, split_name
        )

    return util.QADataset(data_encodings, train=(split_name in ["train", "finetune"])), dataset_dict


def main(rank, world_size, args):
    assert args.model_type in ["distilbert", "moe", "switch_transformer"], "model must be either distilbert, moe, or switch_transformer"
    # define parser and arguments
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"rank {rank}, world_size {world_size}")
    device = rank if world_size > 1 else  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    util.set_seed(args.seed)
    if rank == 0:
        run = wandb.init(project="robustqa", entity="cs224n-robustqa")
    else:
        run = None

    if args.model_type == "distilbert":
        model = DistilBertForQuestionAnswering.from_pretrained(
            "distilbert-base-uncased").to(rank)
    elif args.model_type == "switch_transformer":
        print("using switch transformer")
        ff = FeedForward(args.dim, args.hidden_dim)
        attn=MultiHeadAttention(8, args.dim, 0.2)
        st_ff = SwitchFeedForward(capacity_factor=1.25,drop_tokens=False, n_experts=args.num_experts, expert=ff, d_model=args.dim, is_scale_prob=True)
        st_layer = SwitchTransformerLayer(d_model=args.dim, attn=attn, feed_forward=st_ff,dropout_prob=0.2)
        model = SwitchTransformer(layer=st_layer, n_layers=8, n_experts=args.num_experts, device=device).to(rank)        
    elif args.model_type =="moe":
        print("Using MoE")
        model = MoE(
            dim=args.dim,
            # increase the experts (# parameters) of your model without increasing computation
            num_experts=args.num_experts,
            # size of hidden dimension in each expert, defaults to 4 * dimension
            hidden_dim=args.hidden_dim,
            activation=nn.LeakyReLU,      # use your preferred activation, will default to GELU
            # in top_2 gating, policy for whether to use a second-place expert
            second_policy_train='random',
            # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
            second_policy_eval='random',
            second_threshold_train=0.2,
            second_threshold_eval=0.2,
            # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_train=1.25,
            capacity_factor_eval=2.,      # capacity_factor_* should be set to a value >=1
            # multiplier on the auxiliary expert balancing auxiliary loss
            loss_coef=1e-2,
            device=device
        ).to(rank)
    else:
        raise ValueError("model_type must be either distilbert, moe, or switch_transformer")
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0 or world_size == 1:
        run.config.update(args)
        run.watch(model)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased')

    if args.do_train:
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = device
        if args.pretrain:
            # the train split will be used for pretraining and the 
            # finetune split will be used for finetuning
            pretrain_dataset, _ = get_dataset(args, tokenizer, 'train', args.num_aug_pretrain)
            train_dataset, _ = get_dataset(args, tokenizer, 'finetune', args.num_aug)
        else:
            # No finetuning dataset, only one train dataset
            train_dataset, _ = get_dataset(args, tokenizer, 'train', args.num_aug)
        log.info("Done loading training dataset")
        sampler = RandomSampler(train_dataset) if world_size == 1 else torch.utils.data.distributed.DistributedSampler(train_dataset)

        if args.pretrain:
            pretrain_sampler = RandomSampler(pretrain_dataset) if world_size == 1 else torch.utils.data.distributed.DistributedSampler(pretrain_dataset)
            pretrain_loader = DataLoader(
                pretrain_dataset,
                batch_size=args.batch_size,
                sampler=pretrain_sampler
            )
        else:
            pretrain_loader = None
        
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler= sampler)
        trainer = Trainer(args, log)
 

        val_loader = None
        val_dict = None
        ood_val_loader = None
        ood_val_dict = None
        test_loader = None
        test_dict = None
        if (rank == 0):
            log.info("Preparing in-domain Validation Data...")
            val_dataset, val_dict = get_dataset(args, tokenizer, 'id_val')
            log.info("Preparing out-of-domain Validation Data...")
            ood_val_dataset, ood_val_dict = get_dataset(
                args, tokenizer, "ood_val"
            )
            val_loader = DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    sampler=SequentialSampler(val_dataset))

            ood_val_loader = DataLoader(ood_val_dataset,
                                        batch_size=args.batch_size,
                                        sampler=SequentialSampler(ood_val_dataset))

            test_dataset, test_dict = get_dataset(args, tokenizer, "test")
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,sampler=SequentialSampler(test_dataset))

        if args.model_type == "distilbert":             
            best_scores = trainer.train(
                model, train_loader, val_loader, val_dict, ood_val_loader, ood_val_dict, rank, world_size)
        elif args.model_type in ["moe", "switch_transformer"]:
            best_scores = trainer.train_moe(
                model, pretrain_loader, train_loader, val_loader, val_dict, ood_val_loader, ood_val_dict, test_loader, test_dict, rank, world_size
            )
        else:
            raise ValueError("model_type must be either distilbert, moe, or switch_transformer")

    if args.do_eval:
        args.device = device
        split_name = 'test'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        # TODO: add loading model for MoE
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(
            f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(
            args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    args = get_train_test_args()
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_dir = util.get_save_dir(args.save_dir, args.run_name) 
    if world_size == 1:
        main(0, 1, args)
    else:
        mp.spawn(
            main,
            args=(world_size, args,),
            nprocs=world_size,
            join=True
        )
