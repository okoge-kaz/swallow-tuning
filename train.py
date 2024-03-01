from dataclasses import dataclass
from typing import Optional, Any

import os
import torch
import torch.distributed as torch_distributed
from datasets import disable_caching, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

disable_caching()


@dataclass
class SFTTrainingArguments:
    model_name_or_path: str

    train_data_path: str
    val_data_path: str

    tokenizer_name_or_path: Optional[str] = None

    use_fast: bool = True
    additional_special_tokens: Optional[list[str]] = None
    max_seq_length: int = 4096

    use_flash_attention_2: bool = False

    def from_pretrained_kwargs(self, training_args):
        if training_args.bf16:
            kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}

        if self.use_flash_attention_2:
            kwargs["attn_implementation"] = "flash_attention_2"
        return kwargs


def set_mpi_env() -> None:
    # open mpi config
    global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def print_rank_0(message) -> None:
    if torch_distributed.get_rank() == 0:
        print(message)


def main() -> None:
    set_mpi_env()

    parser = HfArgumentParser(
        dataclass_types=(TrainingArguments, SFTTrainingArguments)  # type: ignore
    )
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    tokenizer_name_or_path: str = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )
    # pad_token追加はエラー
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    train_dataset = load_dataset(
        "json", data_files=sft_training_args.train_data_path, split="train"
    )
    eval_dataset = load_dataset(
        "json", data_files=sft_training_args.val_data_path, split="train"
    )
    print_rank_0(message="dataset load done")

    # dataset
    SYSTEM_PROMPT = [
        {"role": "system", "text": "あなたは誠実で優秀な日本人のアシスタントです。"}
    ]

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            prompt: str = tokenizer.apply_chat_template(
                conversation=SYSTEM_PROMPT + example["input"][i],  # type: ignore
                tokenize=False
            )
            # <s> が余計につくので prompt[3:]
            # output</s> : eos_tokenを最後につける。
            text: str = prompt[3:] + example["output"][i] + tokenizer.eos_token

            output_texts.append(text)
        return output_texts

    response_template = "[/INST]"
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer
    )

    print_rank_0(message="model load start")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
        **kwargs,
    )
    print_rank_0(message="model load end")

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=sft_training_args.max_seq_length,
        args=training_args,
    )

    trainer.train()  # type: ignore

    trainer.save_model()


if __name__ == "__main__":
    main()
