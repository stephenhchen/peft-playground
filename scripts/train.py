import hydra
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          default_data_collator,
                          LlamaForCausalLM,
                          Trainer,
                          TrainingArguments)

from peft_play.data import CommonSenseDataset
from peft_play.utils import get_hf_token


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):

    print("All config:", cfg)

    # Setup
    hf_token = get_hf_token()
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                             inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=.1)
     
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    dataset = CommonSenseDataset(cfg.finetune_dataset.path, tokenizer)

    # model = AutoModelForCausalLM.from_pretrained(cfg.model.name, token=hf_token)
    model = LlamaForCausalLM.from_pretrained(cfg.model.name, token=hf_token)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Add special padding token
    special_tokens = {"pad_token": "[PAD]"}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        logging_dir=cfg.training.logging_dir,
        logging_steps=cfg.training.logging_steps,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        fp16=cfg.training.fp16,
        learning_rate=cfg.training.learning_rate,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    trainer.train()

if __name__ == "__main__":
    train()