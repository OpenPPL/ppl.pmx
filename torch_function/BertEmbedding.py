import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class BertEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_ids: torch.Value, word_weight: torch.Value,
                 token_type_weight: torch.Value, position_weight: torch.Value,
                 position_embedding_type: str='absolute',
                 token_type_ids: Optional[torch.LongTensor] = None):
        if token_type_ids:
            output = g.op('opmx::BertEmbedding',
                          input_ids, word_weight,
                          token_type_weight, position_weight,
                          position_embedding_type_s = position_embedding_type,
                          token_type_ids = token_type_ids)
        else:
            output = g.op('opmx::BertEmbedding',
                          input_ids, word_weight,
                          token_type_weight, position_weight,
                          position_embedding_type_s = position_embedding_type)
        return output.setTypeAs(word_weight)


    @staticmethod
    def forward(self, input_ids: torch.Value, word_weight: torch.Value,
                token_type_weight: torch.Value, position_weight: torch.Value,
                position_embedding_type: str='absolute',
                token_type_ids: Optional[torch.LongTensor] = None,):

        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if torch.onnx.is_in_onnx_export():
            output = torch.zeros([input_shape[0], input_shape[1], word_weight.shape[1]]).to(input_ids.device)
            return output.type_as(word_weight)
        else:
            position_ids = torch.arange(seq_length).expand((1, -1)).to(position_weight.device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

            input_embeds = F.embedding(input_ids, word_weight)
            token_type_embeds = F.embedding(token_type_ids, token_type_weight)
            embeddings = input_embeds + token_type_embeds
            if position_embedding_type == "absolute":
                position_embeds = F.embedding(position_ids, position_weight)
                embeddings += position_embeds

            return embeddings.type_as(word_weight)


def bert_embedding(input_ids: torch.Value, word_weight: torch.Value,
                     token_type_weight: torch.Value, position_weight: torch.Value,
                     position_embedding_type: str='absolute',
                     token_type_ids: Optional[torch.LongTensor] = None,) -> torch.Tensor:
    return BertEmbedding.apply(input_ids, word_weight, token_type_weight, position_weight,
                               position_embedding_type, token_type_ids)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            vocab_size: int,
            max_position_embeddings: int,
            type_vocab_size: int,
            position_embedding_type: str='absolute'):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.position_embedding_type = position_embedding_type

            self.word_weight = nn.Parameter(torch.randn([self.vocab_size, self.hidden_dim]))
            self.token_type_weight  = nn.Parameter(torch.randn([self.type_vocab_size, self.hidden_dim]))
            self.position_weight = nn.Parameter(torch.randn([self.max_position_embeddings, self.hidden_dim]))

        def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.LongTensor] = None):
            return bert_embedding(input_ids, self.word_weight, self.token_type_weight, self.position_weight, self.position_embedding_type, token_type_ids)

    hidden_dim, vocab_size, max_position_embeddings, type_vocab_size = 512, 21128, 8192, 2
    test_op1 = TestModule1(hidden_dim, vocab_size, max_position_embeddings, type_vocab_size)
    input_ids = torch.tensor([[ 101, 3416,  891, 3144, 2945,  118,  122,  102 ]], dtype=torch.int64)

    out = test_op1.forward(input_ids)
    import ipdb;ipdb.set_trace()
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1,  (input_ids), "bert_embedding.onnx", opset_version=11)
    print (model_str1)

    torch.onnx.export(
         test_op1,
         (input_ids),
         "bert_embedding.onnx",
         input_names=['pixel_values'],
         output_names=['vision_embeddings'],
         opset_version=11,
     )
