import torch


class MoeSelect(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g: torch._C.Graph, X: torch.Value, scores: torch.Value,
        num_experts: int, num_experts_per_token: int,
        num_expert_groups: int = 1, num_groups_per_token: int = 1,
        gating_scaling_factor: float = 1.0,
        gating_normalize_prob: bool = False,
        gating_method: str='greedy'):

        X_expand_permute, expert_weights, invert_permutation, expert_offset = (
            g.op("opmx::MoeSelect", X, scores, 
                num_experts_i=num_experts,
                num_experts_per_token_i=num_experts_per_token,
                num_expert_groups_i=num_expert_groups,
                num_groups_per_token_i=num_groups_per_token,
                gating_scaling_factor_f=gating_scaling_factor,
                gating_normalize_prob_i=gating_normalize_prob,
                gating_method_s=gating_method,
                outputs = 4)
        )
        
        return X_expand_permute, expert_weights, invert_permutation, expert_offset


    @staticmethod
    def forward(self, X: torch.Tensor, scores: torch.Tensor,
                num_experts: int, num_experts_per_token: int,
                num_expert_groups: int = 1, num_groups_per_token: int = 1,
                gating_scaling_factor: float = 1.0,
                gating_normalize_prob: bool = False,
                gating_method: str='greedy'):
        # X: [*, hidden_dim]
        # scores: [*, n_experts]
        # X_expand_permute: [*, num_experts_per_token, hidden_dim]
        # expert_weights: [*, num_experts_per_token]
        # invert_permutation: [*, num_experts_per_token]
        # expert_offset: [num_experts + 1]

        assert 'greedy' in gating_method and len('greedy') == len(gating_method)

        if torch.onnx.is_in_onnx_export():
            X_expand_permute = torch.zeros(*X.shape[:-1], num_experts_per_token, X.shape[-1], dtype=X.dtype).to(X.device)
            expert_offset = torch.zeros(num_experts + 1, dtype=int).to(X.device)
            invert_permutation = torch.zeros(*X.shape[:-1], num_experts_per_token, dtype=int).to(X.device)
            expert_weights = torch.zeros(*X.shape[:-1], num_experts_per_token, dtype=X.dtype).to(X.device)
            return X_expand_permute, invert_permutation, expert_offset, expert_weights
        else:
            origin_shape = X.shape
            X_expand_permute = X.view(-1, X.shape[-1])

            _scores = scores.softmax(dim=-1, dtype=torch.float32)
            if 'greedy' in gating_method and len('greedy') == len(gating_method):
                expert_weights, expert_indices = torch.topk(_scores, num_experts_per_token, dim=-1)
            if num_experts_per_token > 1 and gating_normalize_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            else:
                topk_weight *= gating_scaling_factor

            flat_expert_indices = expert_indices.view(-1)   # (seqlen * num_experts_per_token)
            
            sorted_expert_indices, permute_token_idx = flat_expert_indices.sort(stable=True)
            X_expand_permute = X_expand_permute.repeat_interleave(num_experts_per_token, dim=0) # (seqlen * num_experts_per_token, hidden_dim)
            
            X_expand_permute = X_expand_permute[permute_token_idx]

            invert_permutation = torch.full_like(permute_token_idx, -1, device=scores.device)
            for i in range(len(permute_token_idx)):
                reidx = permute_token_idx[i]
                invert_permutation[reidx] = i

            expert_offset = torch.full((num_experts + 1,), -1, device=scores.device)
            ptr = 0
            for i in range(num_experts):
                while(ptr < len(sorted_expert_indices) and sorted_expert_indices[ptr] < i):
                    ptr += 1
                expert_offset[i] = ptr
            expert_offset[num_experts] = X_expand_permute.size(0)
            X_expand_permute = X_expand_permute.view(*origin_shape[:-1], num_experts_per_token, -1)
            invert_permutation = invert_permutation.view(*origin_shape[:-1], num_experts_per_token)

            return X_expand_permute, expert_weights, invert_permutation, expert_offset 


def moe_select(X: torch.Tensor, scores: torch.Tensor,
                num_experts: int, num_experts_per_token: int,
                num_expert_groups: int = 1, num_groups_per_token: int = 1,
                gating_scaling_factor: float = 1.0,
                gating_normalize_prob: bool = False,
                gating_method: str='greedy'):

    return MoeSelect.apply(X, scores, num_experts, num_experts_per_token,
                           num_expert_groups, num_groups_per_token,
                           gating_scaling_factor,
                           gating_normalize_prob,
                           gating_method)


if __name__ == "__main__":
    class TestModule(torch.nn.Module):
        def __init__(self, num_experts: int, num_experts_per_token: int) -> None:
            super().__init__()
            self.num_experts = num_experts
            self.num_experts_per_token = num_experts_per_token


        def forward(self, X: torch.Tensor, scores: torch.Tensor):
            return moe_select(X, scores, self.num_experts, self.num_experts_per_token)


    num_experts, num_experts_per_token = 8, 2
    test_op = TestModule(num_experts, num_experts_per_token)
    X = torch.randn(8, 4096)
    scores = torch.randn(8, num_experts)
    
    
    X_expand_permute, expert_weights, invert_permutation, expert_offset = test_op(X, scores)
    print(expert_offset)
    
    model_str = torch.onnx.export_to_pretty_string(
        test_op, (X, scores), "MoeSelect.onnx", opset_version=11)
    
    print(model_str)
