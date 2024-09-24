# LLaVA: Bridging Language and Vision in AI

## Table of Contents
- [Introduction](#introduction)
- [How LLaVA Works](#how-llava-works)
- [Why LLaVA Stands Out](#why-llava-stands-out)
- [Fine-Tuning LLaVA](#fine-tuning-llava)
  - [Hardware Setup](#hardware-setup)
  - [Data Labeling with UBIAI](#data-labeling-with-ubiai)
  - [Fine-Tuning Process](#fine-tuning-process)
- [Conclusion](#conclusion)

## Introduction

LLaVA (Large Language and Vision Assistant) represents a significant advancement in multimodal AI, combining language processing with visual understanding. This README provides an overview of LLaVA, its architecture, advantages, and the process of fine-tuning for specific tasks.

## How LLaVA Works

LLaVA's architecture integrates:
- A vision encoder (CLIP ViT-L/14) for feature extraction from images
- A Large Language Model (Vicuna, an enhanced version of LLaMA) for language processing

The training process involves two phases:
1. Aligning visual aspects with language using image-text pairs
2. Visual instruction training for complex task handling

## Why LLaVA Stands Out

- LLaVA 1.5 incorporates a multi-layer perceptron (MLP) for enhanced language-vision interaction
- Utilizes task-oriented data from academic sources
- Open-source alternative to proprietary models like GPT-4 Vision
- Cost-effective and scalable
- Strong performance in multimodal benchmarks

## Fine-Tuning LLaVA

### Hardware Setup

For fine-tuning LLaVA-v1.5-13B:
- **GPUs**: NVIDIA A100 or V100 recommended
- **Memory**: 40-80GB GPU memory
- **Parallelism**: Multiple GPUs for reduced training time
- **Storage**: Ample space for model, datasets, and checkpoints

### Data Labeling with UBIAI

- Utilize [UBIAI](https://ubiai.tools) for precise data labeling
- Custom tags: "QUESTION," "ANSWER," and linking tags
- Conversion script provided for transforming UBIAI annotations to LLaVA format

### Fine-Tuning Process

```python
# Set up environment
!pip install -q transformers==4.31.0 accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 einops==0.6.1 flash-attn==2.0.8 xformers==0.0.20 gradio==3.40.1 torchvision sentencepiece

# Load pre-trained model
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = 'liuhaotian/llava-v1.5-7b'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False)

# Configure fine-tuning
!deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /path/to/your/data.json \
    --image_folder /path/to/your/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Merge LoRA weights
!python scripts/merge_lora_weights.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --lora-path ./checkpoints/llava-v1.5-7b-finetune \
    --output-path ./checkpoints/llava-v1.5-7b-finetune-merged

# Evaluate fine-tuned model
model_path = "./checkpoints/llava-v1.5-7b-finetune-merged"
prompt = "Describe this image in detail."
image_file = "path/to/your/image.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "query": prompt,
    "image_file": image_file,
    "conv_mode": None,
    "temperature": 0.2,
    "max_new_tokens": 512,
    "load_8bit": False,
    "load_4bit": False,
})()

eval_model(args)
```

## Conclusion

LLaVA represents a significant advancement in multimodal AI, combining language and vision capabilities. The fine-tuning process allows for customization to specific tasks, enhancing its versatility. This README and the associated Colab tutorial provide a starting point for AI enthusiasts to explore and experiment with LLaVA, contributing to the ongoing development of multimodal AI technologies.
