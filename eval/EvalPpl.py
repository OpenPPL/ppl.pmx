import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset


def evaluate_perplexity(generator, tokenizer):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to("cuda")

    seqlen = 2048
    n_samples = data.numel() // seqlen

    nlls = []


    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to("cuda")
            batch = batch[0]
            with torch.no_grad():
                # ppl_forward args
                ## attn_mask
                attn_mask = torch.empty(0, dtype=torch.float16)
                ## seqstarts
                seqstarts = torch.zeros(2, dtype=torch.int64)
                token_len = len(batch)
                seqstarts[1:] = torch.tensor(token_len, dtype=torch.int64)
                seqstarts = seqstarts.cumsum(0).cuda()
                ## kvlens
                kvstarts = torch.zeros(2, dtype=torch.int64)
                kvlens = [token_len]
                kvstarts[1:] = torch.tensor(kvlens, dtype=torch.int64)
                kvstarts = kvstarts.cumsum(0).cuda()
                ## cachestarts
                cachestarts = torch.tensor([0], dtype=torch.int64).cuda()
                ## decoding_batches
                decoding_batches = torch.tensor([0])
                ## start_pos
                start_pos = torch.tensor([0], dtype=torch.int64).cuda()
                ## max_seqlen
                max_seqlen = torch.tensor([token_len])
                ## max_kvlen
                max_kvlen = torch.tensor([token_len])
                ## kvcache
                total_cache_len = token_len + seqlen
                num_layers = generator.model.params.num_layers
                num_local_kv_heads = generator.model.params.num_kv_heads
                cache_prefix_shape = (total_cache_len, num_layers, 2, num_local_kv_heads)
                head_dim = generator.model.params.hidden_dim // generator.model.params.num_heads
                scale_head_dim = head_dim // generator.model.params.cache_quant_group
                kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.float16).cuda()
                kv_scale = torch.zeros(cache_prefix_shape + (scale_head_dim,), dtype=torch.float16).cuda()
                
                # print(attn_mask, seqstarts, kvstarts, cachestarts, decoding_batches, start_pos, max_seqlen, max_kvlen)
                logits = generator.model.logit_forward(batch, attn_mask, seqstarts, kvstarts,
                                                       cachestarts, decoding_batches, start_pos,
                                                       max_seqlen, max_kvlen, kv_cache, kv_scale)


            shift_logits = logits[:-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]

            loss_fct = nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float()
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()
