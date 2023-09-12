import fire
import json

from pathlib import Path

from Tokenizer import Tokenizer
import Params


def main(
    ckpt_dir: str,
    tokenizer_path: str
):
    tokenizer = Tokenizer(model_path=tokenizer_path)

    params = Params.load(Path(ckpt_dir) / "params.json")
    params.vocab_size = tokenizer.vocab_size()

    pmx_params = Params.cvt_model_args(params)

    with open(Path(ckpt_dir) / "pmx_params.json", "w") as f:
        json.dump(pmx_params.__dict__, f)


if __name__ == "__main__":
    fire.Fire(main)
