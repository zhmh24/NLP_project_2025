import argparse
import random
import json

random.seed(42)

def split_jsonl(input_path, output_path1, output_path2, num_lines1):
    """
    Randomly split a JSONL file into two files.

    Args:
        input_path (str): Path to the input JSONL file.
        output_path1 (str): Path to the first output JSONL file.
        output_path2 (str): Path to the second output JSONL file.
        num_lines1 (int): Number of lines to include in the first output file.
    """
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Split the lines into two parts
    lines1 = lines[:num_lines1]
    lines2 = lines[num_lines1:]

    # Write the first part to output_path1
    with open(output_path1, 'w') as outfile1:
        outfile1.writelines(lines1)

    # Write the second part to output_path2
    with open(output_path2, 'w') as outfile2:
        outfile2.writelines(lines2)


def main():
    parser = argparse.ArgumentParser(description="Randomly split a JSONL file into two parts.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("-o1", "--output1", type=str, required=True, help="Path to the first output JSONL file.")
    parser.add_argument("-o2", "--output2", type=str, required=True, help="Path to the second output JSONL file.")
    parser.add_argument("-n", "--num-lines1", type=int, required=True, help="Number of lines for the first output file.")
    args = parser.parse_args()

    split_jsonl(args.input, args.output1, args.output2, args.num_lines1)


if __name__ == "__main__":
    main()
