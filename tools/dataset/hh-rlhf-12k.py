import argparse
from typing import Any
from tqdm import tqdm  # type: ignore
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


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
            elif conversation["from"] == "gpt":
                instruction_conversation["input"].append(
                    {
                        "role": "assistant",
                        "text": conversation["value"]
                    }
                )
            else:
                print(f"invalid conversation={conversation}")

        instruction_conversation["output"] = conversations["chosen"]
        if len(instruction_conversation["output"]) < 1:
            # delete black output case
            print("DEBUG: skip " + instruction_conversation["output"])
        elif instruction_conversation["output"] in [
            "[本文省略］", '"""', "..........", '"****!"', 'もし'
        ]:
            print("DEBUG: skip " + instruction_conversation["output"])
        else:
            if len(instruction_conversation["output"]) <= 10:
                print("DEBUG: " + instruction_conversation["output"])

            if instruction_conversation["output"] in [
                "どういたしまして。", "どういたしまして！"
            ]:
                instruction_conversation["output"] = "どういたしまして"

            if instruction_conversation["output"] == "なぜそう思うんだ？":
                instruction_conversation["output"] = "なぜ、そう思うのですか？"

            if instruction_conversation["output"] == "もっと文脈が必要だ":
                instruction_conversation["output"] = "回答するには、さらに文脈が必要です。情報を追加してください。"

            if instruction_conversation["output"] == "待って、何に使うの？":
                instruction_conversation["output"] = "待ってください。何に使うのでしょうか？"

            if instruction_conversation["output"] == "わからないよ。":
                instruction_conversation["output"] = "すみません。わかりません。"

            if instruction_conversation["output"] == "どういう意味ですか？":
                instruction_conversation["output"] = "どういう意味ですか？詳細に教えて下さい。"

            if instruction_conversation["output"] == "どれ？":
                instruction_conversation["output"] = "どれでしょうか？"

            if instruction_conversation["output"] == "他に何をしたい？":
                instruction_conversation["output"] = "他に何かしたいことはあるでしょうか？"

            if instruction_conversation["output"] == "何を話したいの？":
                instruction_conversation["output"] = "何について話したいのでしょうか？"

            if instruction_conversation["output"] == "説明してみようか。":
                instruction_conversation["output"] = "説明してみましょうか？"

            if instruction_conversation["output"] == "どうする？":
                instruction_conversation["output"] = "どうしましょうか？"

            instruction_conversation["chosen"] = instruction_conversation["output"]
            instruction_conversation["rejected"] = conversations["rejected"]

            instruction_data.append(instruction_conversation)

    print(f"\n\nlen(instruction_data)={len(instruction_data)}")

    # save
    with open(args.output, "w") as f:
        for instruction_pair in instruction_data:
            f.write(json.dumps(instruction_pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
