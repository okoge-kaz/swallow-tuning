import argparse
from typing import Any
from tqdm import tqdm  # type: ignore
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)

    return parser.parse_args()


def is_invalid_input(conversation: dict) -> bool:
    if "input" in conversation.keys():
        if len(conversation["input"]) == 0:
            return True
        elif len(conversation["input"]) == 1:
            if conversation["input"][0]["text"] == "":
                return True
        elif any(len(chat["text"]) == 0 for chat in conversation["input"]):
            return True
        return False
    else:
        return True


def main() -> None:
    args = arg_parse()
    duplication_count: int = 0

    jsonl_data: list = []
    with open(args.input, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))
    print(f"loaded dataset size={len(jsonl_data)}")

    instruction_data: list[dict[str, Any]] = []

    for conversation in tqdm(jsonl_data):
        if is_invalid_input(conversation=conversation):
            print(f"DEBUG: {conversation}")
            continue
        if len(conversation["output"]) <= 0:
            print("output is 0 len")
            continue
        if conversation in instruction_data:
            duplication_count += 1
            continue
        instruction_data.append(conversation)

    print(f"\n\nlen(instruction_data)={len(instruction_data)}")
    print(f"duplication count={duplication_count}")


if __name__ == "__main__":
    main()
