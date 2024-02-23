import argparse
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("sft-trainers")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    parquet_dataset = load_dataset(args.input)
    # convert jsonl
    parquet_dataset['train'].to_json(args.output, orient='records', lines=True)  # type: ignore


if __name__ == "__main__":
    main()
