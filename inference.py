# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import json
import os
import numpy as np
import struct

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--data_file', default=None, type=str,help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.3,
        max_new_tokens=400
    )

    # The prompt template below is taken from llama.cpp
    # and is slightly different from the one used in training.
    # But we find it gives better results
    prompt_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    )

    sample_data = ["乙肝和丙肝的区别？"]

    def generate_prompt(instruction, input=None):
        if input:
            instruction = instruction + '\n' + input
        return prompt_input.format_map({'instruction': instruction})

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )

    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()
    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    model.eval()

    with torch.no_grad():
            print("Start inference with instruction mode.")

            print('=' * 85)
            print("+ 该模式下仅支持单轮问答，无多轮对话能力。\n"
                  "+ 如要进行多轮对话，请使用llama.cpp或llamachat工具。")
            print('-' * 85)
            print("+ This mode only supports single-turn QA.\n"
                  "+ If you want to experience multi-turn dialogue, please use llama.cpp or llamachat.")
            print('=' * 85)

            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip()) == 0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text, return_tensors="pt")
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                
                compute = model(inputs["input_ids"].to(device)).logits
                # binary_representation = [format(np.float32(item).tobytes(),'08b') for item in compute.cpu().numpy()]
                cpu_tensor = compute
                # 将 CPU 上的 Tensor 转换为 NumPy 数组
                numpy_array = cpu_tensor.cpu().numpy().tobytes().hex()
                # print(numpy_array)
                # binary_representation = [format(np.float32(item).view(np.int32), '032b') for item in numpy_array]
                binary_representation = bin(int(numpy_array,16))[2:]
                print(binary_representation[:8]+"..."+binary_representation[-8:])
                print(compute.shape)
                # print(binary_representation)
                s = generation_output[0]
                
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("### Response:")[1].strip()
                else:
                    response = output

                print(f"Input_tokenizer: {inputs['input_ids']}\n")
                print("Response: ", response)
                print(f"Response_tokenizer: {s}\n")
                print("\n")       

if __name__ == '__main__':
    main()
