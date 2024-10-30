import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# model_name = "/mnt/llm/DeepSeekV2/DeepSeek-V2-Lite"
model_name = "<your_path_to_deepseek_v2>"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda()
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "I believe the meaning of life is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100, )

print(f"======\n in_text: {text}\n======\n")
print(f"======\n in_tok: {inputs['input_ids']}\n======\n")

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"======\n out_text: {result}\n======\n")
print(f"======\n out_tok: {outputs[0]}\n======\n")
