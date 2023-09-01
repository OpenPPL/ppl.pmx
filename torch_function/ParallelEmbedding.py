import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ParallelEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, ids: torch.Value, W: torch.Value, proc_group: torch.Value,
        num_embeddings: int, embedding_dim: int, padding_idx: int = -1,
        max_norm: float = 0, norm_type: float = 2):
        output = g.op("pmx::ParallelEmbedding", ids, W,
                num_embeddings_i = num_embeddings,
                embedding_dims_i = embedding_dim,
                padding_idx_i = padding_idx,
                max_norm_f = max_norm,
                norm_type_f = norm_type)
        return output


    @staticmethod
    def forward(
        self, ids: torch.Tensor, W: torch.Tensor, proc_group: dist.ProcessGroup,
        num_embeddings: int, embedding_dim: int, padding_idx: int = -1,
        max_norm: float = 0, norm_type: float = 2):
        if torch.onnx.is_in_onnx_export():
            output_parallel = torch.zeros(*ids.shape, embedding_dim, dtype=W.dtype).to(W.device)
            if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                last_dim = output_parallel.dim() - 1
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                tensor_list = [torch.zeros_like(output_parallel) for _ in range(world_size)]
                tensor_list[rank] = output_parallel
                output = torch.cat(tensor_list, dim=last_dim).contiguous()
            else:
                output = output_parallel
            return output
        else:
            output_parallel = F.embedding(ids, W,
                None if padding_idx == -1 else padding_idx,
                None if max_norm == 0 else max_norm,
                norm_type, False, False)
            # All-gather across the partitions.
            if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                last_dim = output_parallel.dim() - 1
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                tensor_list = [torch.empty_like(output_parallel) for _ in range(world_size)]
                tensor_list[rank] = output_parallel
                torch.distributed.all_gather(tensor_list, output_parallel, group=proc_group)
                output = torch.cat(tensor_list, dim=last_dim).contiguous()
            else:
                output = output_parallel
            return output


def parallel_embedding(
        ids: torch.Tensor, W: torch.Tensor, proc_group: dist.ProcessGroup,
        num_embeddings: int, embedding_dim: int, padding_idx: int = -1,
        max_norm: float = 0, norm_type: float = 2) -> torch.Tensor:
    return ParallelEmbedding.apply(
                ids, W, proc_group,
                num_embeddings, embedding_dim,
                padding_idx, max_norm,
                norm_type)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            proc_group: dist.ProcessGroup,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int = -1,
            max_norm: float = 0,
            norm_type: float = 2) -> None:
            super().__init__()

            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.proc_group = proc_group

            world_size = 1 if proc_group is None else proc_group.size()
            assert embedding_dim % world_size == 0, "{} is not divisible by {}".format(embedding_dim, world_size)

            self.embedding_dim_per_partition = embedding_dim // world_size

            self.weight = nn.Parameter(torch.ones(self.num_embeddings, self.embedding_dim_per_partition))


        def forward(self, ids: torch.Tensor):
            return parallel_embedding(
                ids, self.weight, self.proc_group,
                self.num_embeddings, self.embedding_dim,
                self.padding_idx, self.max_norm,
                self.norm_type)


    test_op1 = TestModule1(None, 1024, 4096)

    input = torch.tensor([0, 1, 2, 3, 4])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "ParallelEmbedding1.onnx", opset_version=11)

    print(model_str1)
