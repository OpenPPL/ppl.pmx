from typing import List
import sys
import os
import torch
import json

from .Model import Transformer, TensorDumper

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelUtils import __Tokenizer__, __TextGenerator__


class LLaMA(__TextGenerator__):
    def __init__(self, model: Transformer):
        self.model = model


    def generate(
        self,
        prompts_ids: List[List[int]],
        eos_id: int,
        pad_id: int,
        max_gen_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[List[int]]:
        def sample_top_p(probs, p):
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
            return next_token

        unprocessed_prompt_tokens_ids = []
        total_prompt_len = 0
        for i, p in enumerate(prompts_ids):
            unprocessed_prompt_tokens_ids.append(p)
            total_prompt_len = total_prompt_len + len(p)

        total_cache_len = total_prompt_len + len(prompts_ids) * max_gen_len
        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // torch.distributed.get_world_size(group=self.model.proc_group)
        num_layers = self.model.params.num_layers

        if self.model.params.cache_layout == 0:
            cache_prefix_shape = (total_cache_len, num_layers, 2, num_local_kv_heads)
        elif self.model.params.cache_layout == 1:
            cache_prefix_shape = (num_layers, total_cache_len, 2, num_local_kv_heads)
        elif self.model.params.cache_layout == 2:
            cache_prefix_shape = (num_layers, 2, total_cache_len, num_local_kv_heads)
        elif self.model.params.cache_layout == 3:
            cache_prefix_shape = (num_layers, 2, num_local_kv_heads, total_cache_len)
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.int8).cuda()
            kv_scale = torch.zeros(cache_prefix_shape + (scale_head_dim,), dtype=torch.float16).cuda()
        else:
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.float16).cuda()
            kv_scale = None

        max_prompt_len = max([len(t) for t in unprocessed_prompt_tokens_ids])
        output_ids = torch.full((len(prompts_ids), max_prompt_len + max_gen_len), pad_id).cuda().long()
        for k, t in enumerate(unprocessed_prompt_tokens_ids):
            output_ids[k, : len(t)] = torch.tensor(t).long()

        allocated_cache_len = 0
        joined_input = 0
        tokens_ids = []
        start_pos = []
        cachestarts = []
        seqlens = []
        current_batches = 0
        processed_batches = 0
        decoding_batches = torch.tensor([0])
        TensorDumper.step = 0
        batch_ids = []
        current_step = []
        tokens_lens = []
        while True:
            # if len(unprocessed_prompt_tokens_ids) > 0:
            while len(unprocessed_prompt_tokens_ids) > 0:
                joined_input += 1
                t = unprocessed_prompt_tokens_ids.pop(0)
                l = len(t)
                tokens_lens.append(l)
                tokens_ids.extend(t)
                start_pos.append(0)
                cachestarts.append(allocated_cache_len)
                seqlens.append(l)
                allocated_cache_len += l + max_gen_len
                batch_ids.append(processed_batches)
                processed_batches += 1
                current_batches += 1
                current_step.append(0)

            kvlens = [a + b for (a, b) in zip(start_pos, seqlens)]
            max_seqlen = torch.tensor([max(seqlens)])
            max_kvlen = torch.tensor([max(kvlens)])
            _seqstarts = torch.zeros(current_batches + 1, dtype=torch.int64)
            _kvstarts = torch.zeros(current_batches + 1, dtype=torch.int64)
            _start_pos = torch.tensor(start_pos, dtype=torch.int64).cuda()
            _tokens_ids = torch.tensor(tokens_ids, dtype=torch.int64).cuda()

            _seqstarts[1:] = torch.tensor(seqlens)
            _kvstarts[1:] = torch.tensor(kvlens)
            _seqstarts = _seqstarts.cumsum(0)
            _kvstarts = _kvstarts.cumsum(0)

            if self.model.params.cache_mode == 0:
                _cachestarts = torch.tensor(cachestarts, dtype=torch.int64).cuda()
            elif self.model.params.cache_mode == 1:
                _cachestarts = torch.zeros(_kvstarts[-1], dtype=torch.int64).cuda()
                for b, position in enumerate(cachestarts):
                    _cachestarts[_kvstarts[b]:_kvstarts[b+1]] = \
                        torch.arange(position, position + kvlens[b], dtype=torch.int64).cuda()
            else:
                raise Exception("unsupported cache_mode: {}".format(self.model.params.cache_mode))

            _seqstarts = _seqstarts.cuda()
            _kvstarts = _kvstarts.cuda()

            attn_mask = torch.empty(0, dtype=torch.float16)
            if self.model.params.auto_causal == False and decoding_batches < current_batches:
                padded_last_dim = (_kvstarts[-1] + 15) // 16 * 16
                attn_mask = torch.zeros((_seqstarts[-1], padded_last_dim), dtype=torch.float16).cuda()
                for b in range(decoding_batches, current_batches):
                    seqbeg = _seqstarts[b]
                    seqend = _seqstarts[b+1]
                    kvbeg = _kvstarts[b]
                    kvend = _kvstarts[b+1]
                    attn_mask[seqbeg:seqend, kvbeg:kvend] = \
                        torch.triu(torch.full_like(attn_mask[seqbeg:seqend, kvbeg:kvend], float("-inf")), diagonal=1)

            logits = self.model.forward(_tokens_ids, attn_mask, _seqstarts, _kvstarts,
                                        _cachestarts, decoding_batches,
                                        _start_pos, max_seqlen, max_kvlen,
                                        kv_cache, kv_scale)
            TensorDumper.step += 1
            start_pos = [a + b for (a, b) in zip(start_pos, seqlens)]
            seqlens = [1 for _ in seqlens]

            if joined_input > 0:
                decoding_batches += joined_input
                joined_input = 0

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            output_ids[batch_ids, start_pos] = next_token
            tokens_ids = next_token.tolist()
            current_step = [a + 1 for a in current_step]

            removed_batch = []
            for k, p in enumerate(current_step):
                if p >= max_gen_len or tokens_ids[k] == eos_id:
                    removed_batch.append(k)
            removed_batch.reverse()
            for p in removed_batch:
                current_step.pop(p)
                tokens_ids.pop(p)
                start_pos.pop(p)
                cachestarts.pop(p)
                seqlens.pop(p)
                batch_ids.pop(p)
                current_batches -= 1
                decoding_batches -= 1
            if len(current_step) == 0:
                break

        response_ids = []
        for i, t in enumerate(output_ids.tolist()):
            # cut to max gen len
            t = t[: len(prompts_ids[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(eos_id)]
            except ValueError:
                pass
            response_ids.append(t)
        return response_ids


    def export(
        self,
        export_path: str,
    ):
        bsz = 4
        total_len = 16

        total_cache_len = bsz * total_len
        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // torch.distributed.get_world_size(group=self.model.proc_group)
        num_layers = self.model.params.num_layers

        if self.model.params.cache_layout == 0:
            cache_prefix_shape = (total_cache_len, num_layers, 2, num_local_kv_heads)
            max_tokenlen_idx = 0
        elif self.model.params.cache_layout == 1:
            cache_prefix_shape = (num_layers, total_cache_len, 2, num_local_kv_heads)
            max_tokenlen_idx = 1
        elif self.model.params.cache_layout == 2:
            cache_prefix_shape = (num_layers, 2, total_cache_len, num_local_kv_heads)
            max_tokenlen_idx = 2
        elif self.model.params.cache_layout == 3:
            cache_prefix_shape = (num_layers, 2, num_local_kv_heads, total_cache_len)
            max_tokenlen_idx = 3
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.int8)
            kv_scale = torch.zeros(cache_prefix_shape + (scale_head_dim,), dtype=torch.float16)
        else:
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.float16)
            kv_scale = None

        seqlen = total_len // 2
        token_ids = torch.ones(bsz * seqlen, dtype=torch.int64)
        start_pos = torch.zeros(bsz, dtype=torch.int64)
        seqstarts = torch.arange(0, seqlen * (bsz + 1), seqlen, dtype=torch.int64)
        kvstarts = torch.arange(0, seqlen * (bsz + 1), seqlen, dtype=torch.int64)
        decoding_batches = torch.tensor([0], dtype=torch.int64)
        max_seqlen = torch.tensor([seqlen])
        attn_mask = torch.empty(0, dtype=torch.float16)

        if self.model.params.cache_mode == 0:
            cachestarts = torch.arange(0, total_len * bsz, total_len, dtype=torch.int64)
            cachestarts_dim_name = 'batch'
        elif self.model.params.cache_mode == 1:
            cachestarts = torch.arange(0, total_len * bsz, dtype=torch.int64)
            cachestarts_dim_name = 'total_kvlen'
        else:
            raise Exception("unsupported cache_mode: {}".format(self.model.params.cache_mode))

        input_names = ["token_ids", "attn_mask", "seqstarts", "kvstarts",
                       "cachestarts", "decoding_batches",
                       "start_pos", "max_seqlen", "max_kvlen",
                       "kv_cache", "kv_scale"]
        output_names = ["logits"]

        dynamic_axes = {
            'token_ids': {
                0:'total_seqlen'
            },
            'seqstarts': {
                0:'batch + 1'
            },
            'kvstarts': {
                0:'batch + 1'
            },
            'cachestarts': {
                0:cachestarts_dim_name
            },
            'start_pos': {
                0:'batch'
            },
            'kv_cache': {
                max_tokenlen_idx: 'max_tokenlen'
            },
            'kv_scale': {
                max_tokenlen_idx: 'max_tokenlen'
            },
            'logits': {
                0: 'batch',
                1: 'vocab_size'
            },
        }

        if self.model.params.cache_quant_bit == 0:
            dynamic_axes.pop('kv_scale')
            input_names.pop()

        local_rank = torch.distributed.get_rank(group=self.model.proc_group)
        model_path = os.path.join(export_path, "model_slice_{}".format(local_rank))

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.onnx.export(
            self.model.cpu(),
            (token_ids, attn_mask, 
             seqstarts, kvstarts,
             cachestarts, decoding_batches,
             start_pos, max_seqlen, max_seqlen,
             kv_cache, kv_scale),
            os.path.join(model_path, "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            opset_version=11,
            dynamic_axes=dynamic_axes)

        if local_rank == 0:
            with open(os.path.join(export_path, "params.json"), "w") as f:
                json.dump(self.model.params.__dict__, f)
