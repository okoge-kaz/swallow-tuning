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
from trl import DPOTrainer

disable_caching()


@dataclass
class DPOTrainingArguments:
    model_name_or_path: str

    train_data_path: str
    val_data_path: str

    tokenizer_name_or_path: Optional[str] = None
    additional_special_tokens: Optional[list[str]] = None
    max_seq_length: int = 4096

    # dpo config
    dpo_beta: float = 0.01
    dpo_loss_type: str = "sigmoid"

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
        dataclass_types=(TrainingArguments, DPOTrainingArguments)  # type: ignore
    )
    training_args, dpo_training_args = parser.parse_args_into_dataclasses()

    tokenizer_name_or_path: str = (
        dpo_training_args.tokenizer_name_or_path or dpo_training_args.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        additional_special_tokens=dpo_training_args.additional_special_tokens,
        trust_remote_code=True,
    )
    # pad_token追加はエラー
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    train_dataset = load_dataset(
        "json", data_files=dpo_training_args.train_data_path, split="train"
    )
    eval_dataset = load_dataset(
        "json", data_files=dpo_training_args.val_data_path, split="train"
    )
    print_rank_0(message="dataset load done")

    def return_prompt_and_responses(samples) -> dict[str, list[str]]:
        prompts: list[str] = []
        chosens: list[str] = []
        rejecteds: list[str] = []

        for conversation, chosen, rejected in zip(
            samples["input"], samples["chosen"], samples["rejected"]
        ):
            SYSTEM_PROMPT = [
                {"role": "system", "text": "あなたは誠実で優秀な日本人のアシスタントです。"}
            ]
            prompt: str = tokenizer.apply_chat_template(
                conversation=SYSTEM_PROMPT + conversation,  # type: ignore
                tokenize=False
            )
            # <s> が余計につくので
            prompt = prompt[3:]
            assert len(prompt) > 0
            assert len(chosen) > 0
            assert len(rejected) > 0

            prompts.append(prompt)
            chosens.append(chosen)
            rejecteds.append(rejected)

        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    # dataset
    train_dataset = train_dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=train_dataset.column_names,  # type: ignore
    )
    eval_dataset = eval_dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=eval_dataset.column_names,  # type: ignore
    )

    print_rank_0(message="model load start")
    kwargs = dpo_training_args.from_pretrained_kwargs(training_args)
    model = AutoModelForCausalLM.from_pretrained(
        dpo_training_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
        **kwargs,
    )
    print_rank_0(message="model load end")

    trainer = DPOTrainer(
        model,
        beta=dpo_training_args.dpo_beta,
        loss_type=dpo_training_args.dpo_loss_type,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        max_length=dpo_training_args.max_seq_length,
        max_prompt_length=dpo_training_args.max_seq_length // 2,
        max_target_length=dpo_training_args.max_seq_length - (dpo_training_args.max_seq_length // 2),
        args=training_args,
    )

    trainer.train()  # type: ignore

    trainer.save_model()


if __name__ == "__main__":
    main()
