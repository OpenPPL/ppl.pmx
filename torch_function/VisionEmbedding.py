import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, pixel_values: torch.Value, cls_emb_weight: torch.Value,
                 patch_emb_weight: torch.Value, pos_emb_weight: torch.Value,
                 hidden_dim: int, image_size: int, patch_size: int):
        output = g.op('pmx::VisionEmbedding',
                      pixel_values, cls_emb_weight,
                      patch_emb_weight, pos_emb_weight,
                      hidden_dim_i=hidden_dim,
                      image_size_i=image_size,
                      patch_size_i=patch_size)
        return output


    @staticmethod
    def forward(self, pixel_values: torch.Value, cls_emb_weight: torch.Value,
                patch_emb_weight: torch.Value, pos_emb_weight: torch.Value,
                hidden_dim: int, image_size: int, patch_size: int):
        if torch.onnx.is_in_onnx_export():
            output = torch.zeros([pixel_values.shape[0], (image_size//patch_size)**2+1, hidden_dim]).to(pixel_values.device)
            return output
        else:
            num_patches = (image_size // patch_size) ** 2
            num_positions = num_patches + 1
            position_ids = torch.arange(num_positions).expand((1, -1))
            batch_size = pixel_values.shape[0]

            patch_embeds = F.conv2d(pixel_values, patch_emb_weight, stride=patch_size) # shape -> [batch_size, hidden_dim, grid, grid]
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # shape -> [batch_size, grid*grid, hidden_dim]
            cls_embeds  = cls_emb_weight.expand(batch_size, 1, -1)
            pos_embeds = F.embedding(position_ids, pos_emb_weight)

            embeddings = torch.cat([cls_embeds, patch_embeds], dim=1) + pos_embeds # shape -> [batch_size, grid*grid + 1, hidden_dim]
            return embeddings


def vision_embedding(pixel_values: torch.Value, cls_emb_weight: torch.Value,
                     patch_emb_weight: torch.Value, pos_emb_weight: torch.Value,
                     hidden_dim: int, image_size: int, patch_size: int) -> torch.Tensor:
    return VisionEmbedding.apply(pixel_values, cls_emb_weight, patch_emb_weight, pos_emb_weight,
                                 hidden_dim, image_size, patch_size)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            image_size: int,
            patch_size: int):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_positions = (self.image_size // self.patch_size) ** 2 + 1

            self.cls_emb_weight = nn.Parameter(torch.randn(self.hidden_dim))
            self.patch_emb_weight  = nn.Parameter(torch.randn([self.hidden_dim, 3, patch_size, patch_size]))
            self.pos_emb_weight = nn.Parameter(torch.randn(self.num_positions, self.hidden_dim))

        def forward(self, pixel_values: torch.Tensor):
            return vision_embedding(pixel_values, self.cls_emb_weight, self.patch_emb_weight, self.pos_emb_weight, self.hidden_dim, self.image_size, self.patch_size)

    hidden_dim, image_size, patch_size = 512, 224, 32
    test_op1 = TestModule1(hidden_dim, image_size, patch_size)
    pixel_values = torch.ones([1, 3, 224, 224])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1,  (pixel_values), "vision_embedding.onnx", opset_version=11)
    print (model_str1)
    #out = test_op1.forward(pixel_values)

    torch.onnx.export(
        test_op1,
        (pixel_values),
        "vision_embedding.onnx",
        input_names=['pixel_values'],
        output_names=['vision_embeddings'],
        opset_version=11,
    )
