#!/usr/bin/env python

import argparse
import glob
import logging
import os
from os.path import join
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy

import gc
import json
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
import time


from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from transformers import MBartTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BartTokenizer
from transformers.modeling_bart import shift_tokens_right
from utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
    freeze_embeds,
    freeze_params,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    use_task_specific_params,
)


# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa


logger = logging.getLogger(__name__)


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "summarization")
        save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.max_target_length,
            "test": self.hparams.max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.dataset_class = (
            Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        )
        self.already_saved_batch = False
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        #for logging unlikelihood loss
        self.num_outputs = 0
        self.num_ul = 0

        #logging loss to plot training curves, a list of dicts
        #each dict contains the average loss (sum/batch size) for each kind of loss function
        #used to determine relative contributions of UL and standard cross entropy
        self.losses = []


    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""
        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if "mask" not in k else v.shape for k, v in batch.items()
        }
        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")
        save_json({k: v.tolist() for k, v in batch.items()}, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def unlikelihood_loss(self, decoder_input_ids, logits, weight_mask, selective_penalty=False):
        """
        decoder_input_ids - (N, s)
        logits      - (N, s, vocab_size)
        weight_mask - (vocab_size,)
        """
        probs = F.softmax(logits, dim=-1)
        neg_probs = 1-probs

        #replace zeros with small positive constant for stability
        neg_probs += (neg_probs==0).float() * 1e-8
        log_neg_probs = torch.log(neg_probs) # (N,s,v)

        #now create attention mask and apply it
        attention_mask = decoder_input_ids.eq(1).eq(0).float()
        attention_mask = attention_mask.unsqueeze(2).expand(-1,-1,logits.shape[2])
        log_neg_probs_masked = log_neg_probs * attention_mask

        #apply weight vector to the log probability tensor
        N,s = logits.size()[:2]
        weight_mask_expanded = weight_mask.unsqueeze(0).unsqueeze(0).expand(N,s,-1)
        weighted_probs = log_neg_probs_masked * weight_mask_expanded

        if selective_penalty:
            indices = torch.argmax(logits, dim=-1)
            indices_mask = F.one_hot(indices, num_classes=logits.shape[-1]) # (N,s,v)
            weighted_probs *= indices_mask

            #now determine the number of tokens to which UL is applied
            count_vec = (weighted_probs != 0).int() # (N,s,v)
            count_vec = torch.sum(count_vec, dim=-1) # (N,s)
            pad_mask = decoder_input_ids.eq(1).eq(0).int()
            count_vec *= pad_mask

            self.num_outputs += pad_mask.sum()
            self.num_ul += count_vec.sum()


        # TODO: take into account batch size (doesn't matter now since N=1)
        return -torch.sum(weighted_probs)


    def print_penalty_set(self, count_vector, i, num):
        for j in range(num):
            print("********************")
            for k in range(count_vector.shape[-1]):
                if count_vector[i,j,k] > 0:
                    print(f"{k}: {count_vector[i,j,k]}")
            print("********************")


    def unlikelihood_loss_token_old(self, decoder_input_ids, tgt_ids, logits, selective_penalty=True):
        """
        decoder_input_ids - (N, s)
        tgt_ids - (N, s)
        logits      - (N, s, vocab_size)
        """
        probs = F.softmax(logits,dim=-1)

        #replace zeros with small positive constant for stability when applying log
        neg_probs = torch.clamp(1-probs, min=1e-8)
        log_neg_probs = torch.log(neg_probs)

        #for each position, produce a count vector for the penalty set
        N,s,v = logits.shape
        count_vector = decoder_input_ids.unsqueeze(-1).expand(N,s,v) # (N,s,v)
        indices = torch.arange(v).expand(N,s,v).cuda()
        count_vector = torch.cumsum((count_vector==indices).float(), dim=1) # (N,s,v) still

        #now remove special tokens: <s> = 0, </s> = 2, <pad> = 1
        special_token_mask = torch.ones(v).cuda()
        special_token_mask[:3] = 0
        special_token_mask = special_token_mask.expand(N,s,v)
        count_vector *= special_token_mask

        #now remove the token at that position (don't want to penalize the target token)
        cur_token_mask = tgt_ids.unsqueeze(-1).expand(N,s,v)
        cur_token_mask = (cur_token_mask != indices).float()
        count_vector *= cur_token_mask

        #mask padded input positions
        mask_pad = decoder_input_ids.eq(1).eq(0).float() # (N,s)
        mask_pad = mask_pad.unsqueeze(-1).expand(N,s,v)
        count_vector *= mask_pad

        if selective_penalty:
            output_indices = torch.argmax(logits, dim=-1)
            output_indices_mask = F.one_hot(output_indices, num_classes=logits.shape[-1]) # (N,s,v)
            count_vector *= output_indices_mask

        #now compute the actual loss
        loss = count_vector * log_neg_probs
 
        # TODO: take into account batch size (doesn't matter now since N=1)
        return -torch.sum(loss)


    def unlikelihood_loss_token(self, decoder_input_ids, tgt_ids, logits):
        """
        decoder_input_ids - (N, s)
        tgt_ids - (N, s)
        logits      - (N, s, v)
        """
        probs = F.softmax(logits,dim=-1)
        target = tgt_ids.view(-1) # (N,s) -> (N*s,)
        padding_idx = 1 # <pad>

        #replace zeros with small positive constant for stability when applying log
        log_neg_probs = torch.log(torch.clamp(1-probs, min=1e-5))
        log_neg_probs = log_neg_probs.view(-1, log_neg_probs.size(-1)) # (N,s,v) -> (Ns, v)

        with torch.no_grad():
            ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0)) # (Ns, Ns)
            ctx_cands_ = ctx_cands.tril(-1) + padding_idx
            ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
            ctx_cands = ctx_cands.tril(-1) + ctx_cands_

            # Don't include the target for that timestep as a negative target.
            ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), padding_idx)
            negative_targets = torch.zeros_like(log_neg_probs).scatter_(1, ctx_cands, 1)

        #now compute the actual loss
        loss = negative_targets * log_neg_probs

        # TODO: take into account batch size (doesn't matter now since N=1)
        return -loss.sum()


    def get_ngrams(self, tokens, n=1):
        ngrams = set()
        index = 0
        while index <= len(tokens)-n:
            ngrams.add(tuple(tokens[index:index+n]))
            index += 1
        return ngrams

    def ngram_repeat_mask(self, src_ids, gen_ids, decoder_input_ids, n):
        """
        src_ids - (N, s')
        gen_ids - (N, s'')
        decoder_input_ids - (N, s)
        """
        mask = torch.zeros_like(decoder_input_ids).float().cuda()
        for i, (s,g) in enumerate(zip(src_ids,gen_ids)):
            seen = self.get_ngrams(s.tolist(), n)
            gl = g.tolist()

            #make sure to leave out start and end token
            for j in range(1,len(gl)-n):
                ng = tuple(gl[j:j+n])
                if ng in seen:
                    mask[i, j:j+n] = 1

        return mask

    def unlikelihood_loss_copying(self, src_ids, decoder_input_ids, logits, n=3):
        """
        src_ids - (N, s')
        decoder_input_ids - (N, s)
        logits - (N, s, v)
        """
        #now compute actual probability distribution for this iteration
        probs = F.softmax(logits,dim=-1)
        neg_probs = torch.clamp(1-probs, min=1e-8)
        log_neg_probs = torch.log(neg_probs)

        N,s,v = logits.shape

        with torch.no_grad():
            #greedy output
            gen_ids,logs = self.model.generate(src_ids, 
                                     max_length=min(s,args.max_target_length),
                                     early_stopping=False, 
                                     num_return_sequences=1, 
                                     decoder_start_token_id=self.model.config.pad_token_id) # (N, s)

            #create mask
            mask = self.ngram_repeat_mask(src_ids, gen_ids, decoder_input_ids, n) # (N, s)

            #mask padded input positions
            mask_pad = decoder_input_ids.eq(1).eq(0).float() # (N,s)
 
            #extend vector gen_ids to have same sequence length as decoder_input_ids
            if gen_ids.shape[1] < s:
                extra = torch.ones((N, s-gen_ids.shape[1]), dtype=torch.long).cuda()
                gen_ids = torch.cat((gen_ids, extra), dim=1) # (N, s)

            gen_ids_mask = F.one_hot(gen_ids, num_classes=v) # (N,s,v)

        log_neg_probs = torch.sum(log_neg_probs * gen_ids_mask, dim=-1) # (N, s)
        loss = torch.sum(log_neg_probs * mask * mask_pad)
        return -loss


    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        if not self.already_saved_batch:  # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )

        batch_size = src_ids.shape[0]
        loss_log = {'ce': loss.item()}

        if self.unlikelihood_training_tokens:
            ul_token_loss = self.unlikelihood_loss_token(decoder_input_ids, tgt_ids, lm_logits)
            ul_token_loss_weighted = ul_token_loss * self.unlikelihood_tokens_alpha

            print("*******************")
            print(f"loss: {loss}")
            print(f"UL token loss: {ul_token_loss_weighted}")
            print("*******************")

            loss_log['ul_token'] = ul_token_loss_weighted.item()/batch_size
            loss += ul_token_loss_weighted


        if self.unlikelihood_training_copy:
            ul_copy_loss = self.unlikelihood_loss_copying(src_ids, decoder_input_ids, lm_logits, self.unlikelihood_copy_n)
            ul_copy_loss_weighted = self.unlikelihood_copy_alpha * ul_copy_loss

            print("*******************")
            print(f"loss: {loss}")
            print(f"UL copy loss: {ul_copy_loss_weighted}")
            print("*******************")
 
            loss_log['ul_copy'] = ul_copy_loss_weighted.item()/batch_size
            loss += ul_copy_loss_weighted

 
        if self.unlikelihood_training:
            ul_loss = self.unlikelihood_loss(decoder_input_ids, lm_logits, self.weight_vector, self.unlikelihood_selective_penalty)
            ul_loss_weighted = ul_loss * self.unlikelihood_alpha

            print("*******************")
            print(f"loss: {loss}")
            print(f"UL loss: {ul_loss_weighted}")
            print("*******************")

            loss_log['ul_logr'] = ul_loss_weighted.item()/batch_size
            loss += ul_loss_weighted

        self.losses.append(loss_log)
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {
           "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()

        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        generated_ids,logs = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )

        ####################################
        ##        Decoding Strategy       ##
        ####################################

        parser.add_argument('--decode_method', choices=['greedy', 'beam', 'nucleus'])
        parser.add_argument("--decode_num_beams", default=5, type=int, required=False, help="")
        parser.add_argument("--decode_p", default=0.9, type=float, required=False, help="")
        parser.add_argument("--decode_state_change_copy", action="store_true", required=False)
        parser.add_argument("--decode_state_change_copy_alpha", default=0.1, type=float, required=False)
        parser.add_argument("--decode_state_change_copy_n", default=3, type=int, required=False)
        parser.add_argument("--decode_ngram_copy_penalty", default=None, type=int, required=False)
        parser.add_argument("--decode_ngram_copy_weight", default=None, type=float, required=False)

        ####################################
        ##  Unlikelihood Loss Parameters  ##
        ####################################

        parser.add_argument("--unlikelihood_training", action="store_true", help="whether to use unlikelihood training")
        parser.add_argument("--unlikelihood_weights_file", default="cochrane", type=str, required=False, 
                            help="The file containing logistic regression weights for use in unlikelihood training")

        parser.add_argument("--unlikelihood_exclude_tokens", default="", type=str, required=False, help="Comma-separated numbers")
        parser.add_argument("--unlikelihood_num_weights", default=100, type=int, required=False, help="The number of weights in unlikelihood training, if -1 use all of them")
        parser.add_argument("--unlikelihood_softmax", action="store_true", help="whether to softmax the token weights in unlikelihood training")
        parser.add_argument("--unlikelihood_temperature", default=1, type=int, help="temperature to use in softmax when normalizing logistic regression weights")
        parser.add_argument("--unlikelihood_selective_penalty", action="store_true", help="whether to use unlikelihood loss only if argmax is the penalty token")
        parser.add_argument("--unlikelihood_alpha", default=10.0, type=float, required=False, help="")

        ##########################################
        ##  Unlikelihood Loss Token Parameters  ##
        ##########################################

        parser.add_argument("--unlikelihood_training_tokens", action="store_true", help="whether to use unlikelihood training at token level")
        parser.add_argument("--unlikelihood_tokens_alpha", default=1.0, type=float, required=False, help="")

        ##########################################
        ##  Unlikelihood Loss Copy Parameters   ##
        ##########################################

        parser.add_argument("--unlikelihood_training_copy", action="store_true", help="whether to use unlikelihood training to penalize copied text")
        parser.add_argument("--unlikelihood_copy_alpha", default=1.0, type=float, required=False, help="")
        parser.add_argument("--unlikelihood_copy_n", default=3, type=int, required=False, help="")

        return parser

def generate_decode_copy(
    model,
    args,
    input_ids
):
    assert input_ids.shape[0]==1, "Ngram repetition penalty only supported for batch size of 1 (for now)"

    alpha = args.decode_state_change_copy_alpha
    n = args.decode_state_change_copy_n

    temp = model
    model = SummarizationModule(args)
    model.model = temp.to('cuda')
    model.model.train()
    input_ids = input_ids.to('cuda')

    # initialize optimizer
    #optimizer = SGD(model.model.parameters(), lr=1.0)
    #optimizer.zero_grad()

    # first generate output normally
    tgt_ids,logs = model.model.generate(input_ids, 
                           do_sample=False,
                           num_beams=1,
                           max_length=1024,
                           early_stopping=False, 
                           num_return_sequences=1, 
                           decoder_start_token_id=model.model.config.pad_token_id)

    # treat generated text as decoder target
    pad_token_id = model.model.config.pad_token_id
    decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
    outputs = model(input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]

    input_ids = input_ids.tolist()
    decoder_input_ids = decoder_input_ids.tolist()
    tgt_ids = tgt_ids.tolist()

    open('input_ids.txt', 'w').write(json.dumps(input_ids))
    open('decoder_input_ids.txt', 'w').write(json.dumps(decoder_input_ids))
    open('tgt_ids.txt', 'w').write(json.dumps(tgt_ids))
    return None

    # apply repetition penalty
    loss = model.unlikelihood_loss_copying(input_ids, decoder_input_ids, lm_logits, n) * alpha
    #loss.backward()
    #optimizer.step()

    # now re-generate output
    if args.decode_method=='greedy':
        gen_ids,logs = model.model.generate(input_ids, 
                                 do_sample=False,
                                 ngram_copy_penalty=args.decode_ngram_copy_penalty,
                                 ngram_copy_weight=args.decode_ngram_copy_weight,
                                 max_length=args.max_target_length, 
                                 early_stopping=False, 
                                 num_return_sequences=1, 
                                 decoder_start_token_id=model.model.config.pad_token_id)
    elif args.decode_method=='beam':
        gen_ids,logs = model.model.generate(input_ids, 
                                 ngram_copy_penalty=args.decode_ngram_copy_penalty,
                                 ngram_copy_weight=args.decode_ngram_copy_weight,
                                 num_beams=args.decode_num_beams,
                                 max_length=args.max_target_length, 
                                 early_stopping=False, 
                                 num_return_sequences=1, 
                                 decoder_start_token_id=model.model.config.pad_token_id)
    else:
        gen_ids,logs = model.model.generate(input_ids,
                                 do_sample=True,
                                 ngram_copy_penalty=args.decode_ngram_copy_penalty,
                                 ngram_copy_weight=args.decode_ngram_copy_weight,
                                 top_p=args.decode_p,
                                 max_length=args.max_target_length, 
                                 early_stopping=False, 
                                 num_return_sequences=1, 
                                 decoder_start_token_id=model.model.config.pad_token_id)

    return gen_ids,logs


def create_weight_vector(fname, model):

    #prepare weights vectors
    weights = []
    with open(fname) as f:
        for line in filter(lambda l: len(l) > 0, f.readlines()):
            index, weight = line.strip().split()

            if int(index) not in model.unlikelihood_exclude_tokens:
                weights.append((int(index), float(weight)))

    num_weights = model.unlikelihood_num_weights
    weights = [w for w in weights if w[1] < 0]

    if num_weights > -1:
        weights = weights[:num_weights]

    #split ids and weights
    ids = [x[0] for x in weights]
    weights = torch.tensor([abs(x[1]) for x in weights])
    return ids,weights


def set_ul_params(model, hparams):

    #parameters for unlikelihood training
    model.unlikelihood_training = hparams.unlikelihood_training

    if not model.unlikelihood_training:
        return

    model.unlikelihood_weights_file = hparams.unlikelihood_weights_file
    model.unlikelihood_num_weights = hparams.unlikelihood_num_weights
    model.unlikelihood_softmax = hparams.unlikelihood_softmax
    model.unlikelihood_temperature = hparams.unlikelihood_temperature
    model.unlikelihood_selective_penalty = hparams.unlikelihood_selective_penalty
    model.unlikelihood_alpha = hparams.unlikelihood_alpha
    model.unlikelihood_exclude_tokens = set([int(i) for i in hparams.unlikelihood_exclude_tokens.split(',')])
    model.vocab_size = model.model.config.vocab_size

    weights = None
    mask = None
    if model.unlikelihood_weights_file == 'cochrane':
        ids, weights = create_weight_vector("data/logr_weights/bart_freq_normalized_ids.txt", model)
    elif model.unlikelihood_weights_file == 'newsela':
        ids, weights = create_weight_vector("data/logr_weights/bart_freq_newsela_ids.txt", model)
    elif model.unlikelihood_weights_file == 'both':
        ids1, weights1 = create_weight_vector("data/logr_weights/bart_freq_normalized_ids.txt", model)
        ids2, weights2 = create_weight_vector("data/logr_weights/bart_freq_newsela_ids.txt", model)
    else:
        raise Exception()

    if model.unlikelihood_weights_file != 'both':

        if model.unlikelihood_softmax:
            weights = F.softmax(weights/model.unlikelihood_temperature, dim=-1)
        else:
            #normalize alpha if not applying softmax
            model.unlikelihood_alpha /= torch.sum(weights).item()

        weight_vector = torch.zeros(model.vocab_size).float()
        for i in range(len(ids)):
            weight_vector[ids[i]] = weights[i]

        model.weight_vector = weight_vector.to('cuda')
        model.binary_weight_mask = (weight_vector != 0).int()

    else:
        ids = sorted(list(set(ids1 + ids2)))
        id_map = {ID: i for i,ID in enumerate(ids)}
        weights = torch.zeros(len(ids))

        for i,ID in enumerate(ids1):
           weights[id_map[ID]] += weights1[i]

        for i,ID in enumerate(ids2):
           weights[id_map[ID]] += weights2[i]

        if model.unlikelihood_softmax:
            weights = F.softmax(weights/model.unlikelihood_temperature, dim=-1)
        else:
            #normalize alpha if not applying softmax
            model.unlikelihood_alpha /= torch.sum(weights).item()

        #now create weight masks
        weight_vector = torch.zeros(model.vocab_size).float()
        for i in range(len(ids)):
            weight_vector[ids[i]] = weights[i]

        model.weight_vector = weight_vector.to('cuda')
        model.binary_weight_mask = (weight_vector != 0).int()


def set_ul_token_params(model, hparams):
    model.unlikelihood_training_tokens = hparams.unlikelihood_training_tokens
    model.unlikelihood_tokens_alpha = hparams.unlikelihood_tokens_alpha

def set_ul_copy_params(model, hparams):
    model.unlikelihood_training_copy = hparams.unlikelihood_training_copy
    model.unlikelihood_copy_alpha = hparams.unlikelihood_copy_alpha
    model.unlikelihood_copy_n = hparams.unlikelihood_copy_n

def main(args, model=None) -> SummarizationModule:

    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    model = SummarizationModule(args)

    #add unlikelihood parameters - with logr weights
    set_ul_params(model, args)

    #add unlikelihood parameters - token-level
    set_ul_token_params(model, args) 

    #add unlikelihood parameters - copying
    set_ul_copy_params(model, args) 

    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False


    trainer = None

    if args.do_train:

        lower_is_better = args.val_metric == "loss"
        save_top_k = args.max_epochs

        trainer: pl.Trainer = generic_train(
            model,
            args,
            logging_callback=Seq2SeqLoggingCallback(),
            checkpoint_callback=get_checkpoint_callback(
                args.output_dir, model.val_metric, save_top_k, lower_is_better
            ),
            early_stopping_callback=es_callback,
            logger=logger,
        )
        pickle_save(model.hparams, model.output_dir / "hparams.pkl")

        #now write loss logs into the same directory
        with open(os.path.join(args.output_dir, 'loss_logs.json'), 'w') as f:
            f.write(json.dumps(model.losses, indent=2))


    if args.do_generate:

        if args.generate_epoch > -1:
            model = BartForConditionalGeneration.from_pretrained(join(args.output_dir, f'best_tfmr-{args.generate_epoch}'))
        else:
            print("********* using fresh model *********")
            args.generate_epoch = 'no-train'
            model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
 
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

        abstracts = list(open(join(args.data_dir, f'{args.generate_input_prefix}.source')).readlines())
        pls = list(open(join(args.data_dir, f'{args.generate_input_prefix}.target')).readlines())
        dois = list(open(join(args.data_dir, f'{args.generate_input_prefix}.doi')).readlines())

        if args.generate_start_index == 'none' and args.generate_end_index != 'none':
            abstracts = abstracts[:int(args.generate_end_index)]
            pls = pls[:int(args.generate_end_index)]
            dois = dois[:int(args.generate_end_index)]
        elif args.generate_start_index != 'none' and args.generate_end_index == 'none':
            abstracts = abstracts[int(args.generate_start_index):]
            pls = pls[int(args.generate_start_index):]
            dois = dois[int(args.generate_start_index):]
        elif args.generate_start_index != 'none' and args.generate_end_index != 'none':
            abstracts = abstracts[int(args.generate_start_index):int(args.generate_end_index)]
            pls = pls[int(args.generate_start_index):int(args.generate_end_index)]
            dois = dois[int(args.generate_start_index):int(args.generate_end_index)]

        abstracts_final = []
        dois_final = []
        pls_final = []
        gen_final = []
 
        batch = tokenizer(abstracts, padding='max_length', max_length=args.max_source_length, truncation=True, return_tensors='pt')
        input_ids = batch['input_ids']

        fname_prefix = f'gen_{args.decode_method}_'
        if args.decode_state_change_copy:
            fname_prefix += f'decode-state-change-copy-{args.decode_state_change_copy_alpha}-{args.decode_state_change_copy_n}_'
        elif args.decode_ngram_copy_penalty is not None and args.decode_ngram_copy_weight is not None:
            fname_prefix += f'copy-{args.decode_ngram_copy_penalty}-{args.decode_ngram_copy_weight}_'
        fname_prefix += f'{args.generate_input_prefix}_{args.generate_epoch}_{args.generate_start_index}-{args.generate_end_index}'

        fname_text = fname_prefix + '_text_only.txt'

        logs_list = []
        for i,d,a,p in zip(range(len(dois)), dois, abstracts, pls):
            ids = input_ids[i]
            logs = None

            if args.decode_state_change_copy:
                gen_ids,logs = generate_decode_copy(model, args, ids.unsqueeze(0))
            elif args.decode_method=='greedy':
                gen_ids,logs = model.generate(ids.unsqueeze(0), 
                                         do_sample=False,
                                         ngram_copy_penalty=args.decode_ngram_copy_penalty,
                                         ngram_copy_weight=args.decode_ngram_copy_weight,
                                         max_length=args.max_target_length, 
                                         early_stopping=False, 
                                         num_return_sequences=1, 
                                         decoder_start_token_id=model.config.pad_token_id)
            elif args.decode_method=='beam':
                gen_ids,logs = model.generate(ids.unsqueeze(0), 
                                         ngram_copy_penalty=args.decode_ngram_copy_penalty,
                                         ngram_copy_weight=args.decode_ngram_copy_weight,
                                         num_beams=args.decode_num_beams,
                                         max_length=args.max_target_length, 
                                         early_stopping=False, 
                                         num_return_sequences=1, 
                                         decoder_start_token_id=model.config.pad_token_id)
            elif args.decode_method=='nucleus':
                gen_ids,logs = model.generate(ids.unsqueeze(0),
                                         do_sample=True,
                                         ngram_copy_penalty=args.decode_ngram_copy_penalty,
                                         ngram_copy_weight=args.decode_ngram_copy_weight,
                                         top_p=args.decode_p,
                                         max_length=args.max_target_length, 
                                         early_stopping=False, 
                                         num_return_sequences=1, 
                                         decoder_start_token_id=model.config.pad_token_id)

            gen_text = tokenizer.decode(gen_ids.squeeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            dois_final.append(dois[i])
            abstracts_final.append(abstracts[i])
            pls_final.append(pls[i])
            gen_final.append(gen_text)

            if logs is not None:
                logs_list.append(logs)

            with open(join(args.output_dir, fname_text), 'a+') as f:
                f.write(gen_text + '\n----------------------------------------\n')
                f.flush()
            
            print(gen_text + '\n----------------------------------------\n')

        output = [{'doi': d, 'abstract': a, 'pls': p, 'gen': g} for d,a,p,g in zip(dois_final, abstracts_final, pls_final, gen_final)]

        fname_json = fname_prefix + '.json'
        open(join(args.output_dir, fname_json), 'w').write(json.dumps(output, indent=2))

        if len(logs_list) > 0:
            fname_logs = fname_prefix + '_log.json'
            open(join(args.output_dir, fname_logs), 'w').write(json.dumps(logs_list, indent=2))

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    main(args)
