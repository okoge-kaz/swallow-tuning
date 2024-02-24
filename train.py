from dataclasses import dataclass
from typing import Optional, Any

import os
import torch
from datasets import disable_caching, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

disable_caching()


@dataclass
class SFTTrainingArguments:
    model_name_or_path: str

    data_files: list[str]
    eval_data_files: Optional[list[str]] = None
    tokenizer_name_or_path: Optional[str] = None

    use_fast: bool = True
    additional_special_tokens: Optional[list[str]] = None
    max_seq_length: int = 4096

    use_flash_attention_2: bool = False

    use_neftune: bool = False
    neftune_noise_alpha: Optional[float] = None

    def from_pretrained_kwargs(self, training_args):
        if training_args.bf16:
            kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs


def load_datasets(data_files: list[str]):
    datasets = []
    for data_file in data_files:
        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]  # type: ignore
        dataset = dataset.select_columns("text")  # type: ignore
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def set_mpi_env() -> None:
    # open mpi config
    global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)


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

    train_dataset = load_datasets(sft_training_args.data_files)
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files)
        training_args.do_eval = True
    else:
        eval_dataset = None

    instruction_ids = tokenizer.encode("\n\n### 指示:\n", add_special_tokens=False)[1:]
    response_ids = tokenizer.encode("\n\n### 応答:\n", add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_ids, response_template=response_ids, tokenizer=tokenizer
    )

    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=sft_training_args.max_seq_length,
        neftune_noise_alpha=sft_training_args.neftune_noise_alpha if sft_training_args.use_neftune else None,
    )

    trainer.train()  # type: ignore

    trainer.save_model()


if __name__ == "__main__":
    main()
