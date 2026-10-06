import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  

import gc, json, random
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

base_model_id = 'meta-llama/Llama-3.2-3B-Instruct'
MAX_SEQ_LEN = 1536   # inchangé (voir « points de vigilance »)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])

tokenizer = AutoTokenizer.from_pretrained(base_model_id, clean_up_tokenization_spaces=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

ASSISTANT_HEADER_IDS = tokenizer.encode(
    '<|start_header_id|>assistant<|end_header_id|>\n\n', add_special_tokens=False)


class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Masque tous les tokens du prompt ; la loss ne porte que sur la complétion."""
    def __init__(self, tokenizer, response_ids):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.response_ids = list(response_ids)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch['labels'].clone()
        tpl = self.response_ids
        n_tpl = len(tpl)
        for i in range(labels.size(0)):
            seq = batch['input_ids'][i].tolist()
            start = None
            for j in range(len(seq) - n_tpl + 1):
                if seq[j:j + n_tpl] == tpl:
                    start = j + n_tpl
                    break
            if start is None:
                labels[i, :] = -100
            else:
                labels[i, :start] = -100
        batch['labels'] = labels
        return batch


collator = CompletionOnlyCollator(tokenizer, ASSISTANT_HEADER_IDS)


def train_lora_adapter(train_dataset, val_dataset, out_dir, seed=42, epochs=5):
    """Un adaptateur QLoRA indépendant, à partir du même modèle de base."""
    set_seed(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={'': 0},            # le worker ne voit qu'UN GPU (CUDA_VISIBLE_DEVICES)
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation='eager',
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, lora_cfg)

    args = SFTConfig(
        output_dir=str(out_dir) + '_checkpoints',
        num_train_epochs=epochs,
        per_device_train_batch_size=1,     # était 2
        per_device_eval_batch_size=1,      # était 2
        gradient_accumulation_steps=4,     # était 2  -> batch effectif = 4 (inchangé)
        eval_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type='cosine',
        logging_steps=10,
        logging_strategy='steps',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        optim='paged_adamw_8bit',
        report_to='none',
        dataset_text_field='text',
        max_length=MAX_SEQ_LEN,
        packing=False,
        seed=seed,
        data_seed=seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
        data_collator=collator,
    )
    print(f'Training seed={seed} -> {out_dir}')
    train_output = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = dict(train_output.metrics)
    metrics['seed'] = seed
    metrics['peak_gpu_gib'] = torch.cuda.max_memory_allocated() / 2**30
    # écrit EN DERNIER : sert de marqueur « entraînement terminé »
    with open(str(out_dir) + '_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Pic mémoire GPU : {metrics['peak_gpu_gib']:.2f} GiB")
    return metrics