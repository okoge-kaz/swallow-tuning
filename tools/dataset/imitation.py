import argparse
from typing import Any
from sympy import true
from tqdm import tqdm  # type: ignore
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def is_invalid_input(conversation: dict) -> bool:
    if "input" in conversation.keys():
        if len(conversation["input"]) == 0:
            return True
        elif len(conversation["input"]) == 1:
            if conversation["input"][0]["text"] == "":
                return True
        elif any(len(role_text["text"]) == 0 for role_text in conversation["input"]):
            return True

        return False
    else:
        return True


def is_invalid_output(conversation: dict) -> bool:
    if "output" in conversation.keys():
        if conversation["output"] == "":
            return True
        if conversation["output"] == "\n":
            return True

    return False


def main() -> None:
    args = arg_parse()

    jsonl_data: list = []
    with open(args.input, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))

    instruction_data: list[dict[str, Any]] = []
    for conversations in tqdm(jsonl_data):
        instruction_conversation: dict = {
            "input": [],
        }

        for conversation in conversations["conversations"]:

            if conversation["from"] == "human":
                instruction_conversation["input"].append(
                    {
                        "role": "user",
                        "text": conversation["value"]
                    }
                )
            elif conversation["from"] == "mixtral":
                instruction_conversation["output"] = conversation["value"]
                instruction_data.append(instruction_conversation)

                instruction_conversation = {
                    "input": instruction_conversation["input"].copy(),
                }
                instruction_conversation["input"].append(
                    {
                        "role": "assistant",
                        "text": conversation["value"]
                    }
                )
            else:
                print(f"invalid conversation={conversation}")

    print(f"\n\nmid len(instruction_data)={len(instruction_data)}\n\n")

    filtered_instruction_data: list = []
    seen = []
    duplicated_count: int = 0

    for instruction in tqdm(instruction_data):
        if is_invalid_input(conversation=instruction):
            print(f"invalid={instruction}")
        elif is_invalid_output(conversation=instruction):
            pass
        else:
            if instruction in seen:
                duplicated_count += 1
                continue
            else:
                filtered_instruction_data.append(instruction)
                seen.append(instruction)

    print(f"\n\nfinal len(instruction_data)={len(filtered_instruction_data)}")
    print(f"duplicated count={duplicated_count}")

    # save
    with open(args.output, "w") as f:
        for instruction_pair in filtered_instruction_data:
            f.write(json.dumps(instruction_pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
