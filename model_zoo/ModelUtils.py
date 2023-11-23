import torch
from typing import List

class __TensorDumper__:
    def __init__(self):
        self.enable_dump = False
        self.dir = "."
        self.step = 0
        self.dump_steps = []


    def dump(self, X: torch.Tensor, name: str):
        if not self.enable_dump:
            return
        
        if len(self.dump_steps) > 0 and self.step not in self.dump_steps:
            return

        if X is None:
            X = torch.empty(0)

        shape_str = "" if X.dim == 0 else str(X.shape[0])
        for d in X.shape[1:]:
            shape_str = shape_str + "_" + str(d)

        type_dict = {
            torch.float: "fp32",
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.int8: "int8",
            torch.int64: "int64",
        }

        filename = "step{}_{}-{}-{}.bin".format(self.step, name, shape_str, type_dict[X.dtype])

        X.cpu().numpy().tofile(self.dir + "/" + filename)


class __Tokenizer__:
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        raise Exception("inferface class is unable to call")

    def decode(self, t: List[int]) -> str:
        raise Exception("inferface class is unable to call")

    def vocab_size(self) -> int:
        raise Exception("inferface class is unable to call")

    def get_bos_id(self) -> int:
        raise Exception("inferface class is unable to call")

    def get_eos_id(self) -> int:
        raise Exception("inferface class is unable to call")
    
    def get_pad_id(self) -> int:
        raise Exception("inferface class is unable to call")


class __TextGenerator__:
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
        raise Exception("inferface class is unable to call")

    def export(
        self,
        export_path: str
    ):
        raise Exception("inferface class is unable to call")
