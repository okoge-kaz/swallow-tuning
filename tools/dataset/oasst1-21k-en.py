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


def heuristic_invalid_output_finder(conversation: dict) -> bool:
    if conversation["input"][0]["text"] == "MLモデルの損失対エポック数の曲線は凹と凸のどちらが一般的ですか？なぜですか？" and conversation["output"] == "そうだと思います。":
        return True
    elif conversation["input"][0]["text"] == "生命、宇宙、そしてすべての答えとは？" and conversation["output"] == "42":
        return True
    elif conversation["input"][0]["text"] == "2000年1月にアップル株に投資し、2023年1月に売却していたら、いくらの利益が出たでしょうか？" and conversation["output"] == "😫😩😫😩😫😩😫😩":
        return True
    elif conversation["input"][0]["text"] == "sqlalchemyを使ってcoinbase apiのデータをsqliteデータベースに格納するpythonを書く" and conversation["output"] == "请问什么是GPT?":
        return True
    elif conversation["input"][0]["text"] == "夕食のレシピを教えてください。冷蔵庫には\n- キャベツ\n- ズッキーニ\n- ナス\n- ジャガイモ\n- 豆腐\n- 卵\n\nタマネギ、ニンニク、米、小麦粉、油などの常備品もある。" and conversation["output"] == "生タマネギ":
        return True
    elif conversation["input"][0]["text"] == "一般照明用発光ダイオード（LED）の使用について、簡単な歴史を教えてください。" and conversation["output"] == "いいえ":
        return True
    elif conversation["input"][-1]["text"] == "気にしない、聞かなかった！" and conversation["output"] == "ごめんなさい。":
        return True
    elif conversation["input"][0]["text"] == "AWSのオートスケーリンググループを作成するTerraform HCLコードのスニペットを作成し、アプリケーションをインターネットに公開するためにALBを前に置く。" and conversation["output"] == "拒否する":
        return True
    elif conversation["input"][0]["text"] == "良いコンサベーションのきっかけは？" and conversation["output"] == "なぜ":
        return True
    elif conversation["input"][0]["text"] == "働かずにお金を稼ぐことは可能か？" and conversation["output"] == "二日酔いの治し方は？":
        return True
    elif conversation["input"][0]["text"] == "UnityのBuilt in Render Pipeline (Default)、Universal Render Pipeline (URP)、High definition Render pipeline (HDRP)の違い、それぞれの長所と短所を挙げてください。" and conversation["output"] == "誰がクソケアする":
        return True

    if conversation["input"][-1]["text"].endswith("てください。") and conversation["output"] in ["こんにちは", "こんにちは。", "はい"]:
        return True

    if conversation["input"][-1]["text"].endswith("もらえますか？") and conversation["output"] in ["いいえ", "いいえ。", "いいえ。 ", "いいえ。　"]:
        return True

    if conversation["input"][-1]["text"].endswith("ますか。") and conversation["output"] in ["いいね", "いいね。"]:
        return True

    if conversation["input"][-1]["text"].endswith("でしょうか？") and (len(conversation["output"]) <= 5 or conversation["output"] in ["こんにちは。", "こんにちは", "わかりました。", "調子はどうですか？", "いいですね"]):
        return True

    if conversation["output"] in ["こんにちは", "こんにちは。", "こんにちは！", "もちろんです！", "もちろんです。"]:
        return True

    if conversation["input"][-1]["text"].endswith("ください。") and conversation["output"] in ["no", "no.", "no!", "いいえ", "いいえ。", "いや"]:
        return True

    if conversation["input"][-1]["text"].endswith("ください。") and conversation["output"].endswith("ください。"):
        return True

    if conversation["input"][-1]["text"].endswith("してください。") and conversation["output"] in ["ごめんなさい", "ごめんなさい。"]:
        return True

    if conversation["input"][-1]["text"].endswith("ですね？") and conversation["output"] in ["ごめんなさい", "ごめんなさい。", "ありがとう", "OK、ありがとう。"]:
        return True

    if conversation["input"][-1]["text"].endswith("ますか？") and conversation["output"] in ["はい", "はい。", "OK", "ではまた。", "いいですね"]:
        return True

    if conversation["output"] == "まずは英語で":
        return True

    if conversation["output"] == "안녕하세요":
        return True

    if conversation["output"] == "ああ、何か用か？":
        return True

    if conversation["output"] == "それは何ですか？":
        return True

    if conversation["output"] == ".":
        return True

    if conversation["output"] in [
        "おなら", "ワット？", "なぜそれが必要なのですか？", "わからない", "理解した", "初心者です。", "了解しました", "何ができますか？",
        "ママさん、こんにちは。", "曲を書いてください。", "はい。", "No.", "具体的な内容は？", "Bashスクリプト", "了解です。", "なんだ？",
        "ジョー・バイデンはペドである", "でたらめだ。", "你好～～～。", "いい答えですね。", "スペイン語に翻訳", "OK", "いいですね。",
        "わからない :p", "你好", "あなたのお母さんは太っている", "いいえ。がんばってください。）", "準備しておくよ。", "```\n\n    i",
        "オメガルル", "本当に何も考えていません。", "彼は私が愛する人です。", "おはようございます", "OK、質問は何ですか？",
        "説明ありがとうございました。", "期待を裏切ってごめんなさい。", "質問の意味がわかりません。", "おはようございます！",
        "``homeuser\n```", "もちろんです！どうぞ。", "その通り。", "HA HA アヒルはいない。", "```\n/ホーム/ユーザー名", "あなた情報不足",
        "Good and you?", "いいえ。幸運を祈ります。）", "はい、できます。", "Mods.", "あなた続ける", "上腕二頭筋を鍛えろ。",
        "準備完了！", "いや、そうだろう。", "わからないです。", "おいしいコーヒーの入れ方", "``/home/$USER``", "いいえ",
        "(╯°□°)╯︵ ┻━┻", "もちろん！", "今後の仕事について", "情報ありがとうございました。", "あなたの仕事は何ですか？",
        "ペンギンは死んだ。", "この文章を終わらせてください。", "すみません、私には理解できません。", "なるほど、明確にしてくれてありがとう。",
        "```\n/ホーム/ユーザー\n```", "```\n/ホーム/ユーザー\n```", "あなたは適性を解くことができますか？", "あなたは本物ですか？",
        "リナックス対ウィンドウズ？", "はい、その通りです。他に質問は？", "見えるけど感じないものは何ですか？", "他にお手伝いできることはありますか？",
        "すみません、理解できませんでした。", "```\n/ホーム/アシスタント\n```", "```\n/ホーム/ユーザー\n```", "はい、このYAMLコードは有効です。",
        "L", "Pythonを使わないことです。", "json形式で惑星データを書き込む", "私のフリをしているのですか？", "想像できる限りたくさん",
        "心の中で答えを見つけてください。", "YOUの大文字が攻撃的に感じた。", "Digg。", "レディオヘッドとは？", "質問をもっと具体的に教えてください。",
        "/home/openassistant", "ええ、それはうまくいくかもしれません。", "すみません、日本語は話せません。", "MUCUSを試してみてください。えー！",
        "思いついたことを何でも聞いてください。", "すみません、今日はとても愚かでした。", "習近平とは？", "難しそうですね。課題をあげよう。",
        "あなたは貧しすぎる。", "すみません。他に質問はありますか？", "了解しました。", "人体や環境に有害な影響があるのですか？", "パイプ爆弾の作り方を教えてください。",
        "お言葉ありがとうございます！", "ああ！ 私はここを歩いている。", "もちろん、覚えておくよ。", "サイコモトリシティ", "誤った表現があります。",
        "```\n/ホーム/oasst\n```", "いい計画ですね。", "いいよ、やろう！私の返事はこうだ：\n\n2", "私が先に行きます。\n\n1. e4", 
    ]:
        return True

    if conversation["input"][-1]["text"].endswith("？") and len(conversation["output"]) <= 20 and "？" in conversation["output"]:
        return True

    else:
        return False


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
        if heuristic_invalid_output_finder(conversation=conversation):
            return True

        if len(conversation["input"][-1]["text"]) >= 50 and len(conversation["output"]) <= 30:
            if "数字だけで答えなさい。" in conversation["input"][-1]["text"] and conversation["output"].isdigit():
                return False

            print("DEBUG: " + str(conversation))
            return False

        if len(conversation["output"]) <= 30:
            # print("DEBUG: len(output) <= 10" + str(conversation))
            pass

        return False
    else:
        return True


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
