import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import json
import os

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
import numpy as np

# _GLOBAL_INTERVENED_TOKENS = set()

#----------------收集工具---------------------
import json
import torch
import torch.nn.functional as F
import os
import numpy as np


def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        key_position: Optional[dict] = None,
        sample_greedy: Optional[bool] = None,
        **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    model_kwargs_cd = model_kwargs.copy()
    input_ids_cd = model_kwargs_cd.get("input_ids_cd", None)  # for instruction cd only
    prompt_length = input_ids.shape[1]
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        #  turn to normal input
        self.model.config.use_fast_v = False
        self.model.reset_fastv()
        use_icd = (model_kwargs.get("input_ids_cd") != None) or (model_kwargs.get("inputs_embeds_cd") != None)
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states for analysis
            vad=False,
            key_position=None,
            Entropy_ids2=input_ids,
        )

        next_token_logits = outputs.logits[:, -1, :]

        ## For contrastive decoding initial
        use_icd = False
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        # model_kwargs_cd = model_kwargs.copy()

        if use_cd:  # for VCD and SID
            if self.model.config.fast_v_attention_rank != None:
                self.model.config.use_fast_v = True
                self.model.reset_fastv()

            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
                key_position=key_position,
                vad=False,
                # logits_processor=logits_processor,
                Entropy_ids2=input_ids
            )
            # key_position['enable_debiasing'] = None/home/kdzlys/sid2/log/llava-1.5/sampling——2到15的0.1JS.jsonl
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            ## cd_comments: pre-process logits from contrastive inputs
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

            JS_PH_alpha = model_kwargs.get("JS_PH_alpha", 2)
            JS_threshold = model_kwargs.get("JS_threshold", 0.15)

            cutoff1 = torch.log(torch.tensor(cd_beta))
            cutoff = cutoff1 + next_token_logits.max(dim=-1, keepdim=True).values
            P = next_token_logits[0].float().cpu().numpy()
            Q = next_token_logits_cd[0].float().cpu().numpy()
            def softmax_with_temperature(logits, temperature, axis=-1):
                logits_scaled = logits / temperature
                logits_scaled = logits_scaled - np.max(logits_scaled, axis=axis, keepdims=True)
                exps = np.exp(logits_scaled)
                probabilities = exps / np.sum(exps, axis=axis, keepdims=True)
                return probabilities

            def cal_JS_div(p_logits, q_logits, temperature=1.0, clip_value=5e-8, axis=-1, normalize_to_01=False):
                p = softmax_with_temperature(p_logits, temperature, axis=axis)
                q = softmax_with_temperature(q_logits, temperature, axis=axis)

                p = np.clip(p, clip_value, 1.0)
                q = np.clip(q, clip_value, 1.0)

                m = 0.5 * (p + q)

                kl_p_m = np.sum(p * np.log(p / m), axis=axis)
                kl_q_m = np.sum(q * np.log(q / m), axis=axis)
                jsd = 0.5 * (kl_p_m + kl_q_m)

                if normalize_to_01:
                    jsd = jsd / np.log(2.0)

                return jsd

            JS = cal_JS_div(P, Q,1.0)
            if JS < JS_threshold:
                diffs = next_token_logits + JS_PH_alpha * next_token_logits_cd
            else:

                diffs = next_token_logits +JS*next_token_logits_cd

            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            if sample_greedy:
                next_tokens = torch.argmax(diffs, dim=-1)
            else:
                cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)


        elif use_icd:
            if model_kwargs.get("input_ids_cd") == None:  # Q-former
                model_inputs_cd = self.prepare_inputs_for_generation_icd(input_ids=input_ids, **model_kwargs_cd)
            else:  # VLM connector
                model_inputs_cd = self.prepare_inputs_for_generation_icd(input_ids=input_ids_cd, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=True,
                key_position=key_position,
                vad=False,
            )
            image_start_pos = key_position.get("image_start") if key_position is not None else None

            next_token_logits_cd = outputs_cd.logits[:, -1, :]


            ## cd_comments: pre-process logits from contrastive inputs 这里可能是由问题的
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

            # version 1  set cutoff for Adaptive Plausibility Constraints
            # probs = nn.functional.softmax(next_token_logits, dim=-1)
            # cutoff = 0.8 * probs.max(dim=-1, keepdim=True).values
            # cutoff = 0.9 * next_token_logits.max(dim=-1, keepdim=True).values

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff1 = torch.log(torch.tensor(cd_beta))
            cutoff = cutoff1 + min(next_token_logits.max(dim=-1, keepdim=True).values,next_token_logits_cd.max(dim=-1, keepdim=True).values)

            diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd

            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            if sample_greedy:  # argmax
                next_tokens = torch.argmax(diffs, dim=-1)
            else:  ## sampling:
                cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

            # for instruction cd, update generated ids, model inputs, and length for next step
            if model_kwargs.get("input_ids_cd") != None:  # VLM connector
                input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)#tensor([[0]], device='cuda:0')
            next_token_scores = logits_warper(input_ids, next_token_scores)

            if sample_greedy:  # argmax
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            else:  ## sampling:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break


    if streamer is not None:
        streamer.end()


    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample