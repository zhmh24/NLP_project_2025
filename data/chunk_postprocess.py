import argparse
import os
import json
import numpy as np
from pathlib import Path
import string

def stage_1(folder_name, output_base_path):
    """
    Stage 1: Map the ids to new ids by merging blocks with the same base name.
    Generate additional outputs: new npy file, meta.json, id_to_snbt mapping, id counts, and id-to-id mapping.
    """
    input_path = f"data/{folder_name}/id_to_snbt.json"
    npy_path = f"data/{folder_name}/block_ids.npy"
    output_path = os.path.join(output_base_path, folder_name, 's1')

    with open(input_path, 'r') as f:
        id_to_snbt = json.load(f)

    # Reverse the mapping to group by base names
    snbt_to_id = {v: int(k) for k, v in id_to_snbt.items()}
    base_name_to_new_id = {}
    new_id_to_base_name = {}
    new_id = 0
    old_to_new_id = {}
    id_counts = {}
    base_name_to_full_snbt = {}

    for snbt, old_id in snbt_to_id.items():
        base_name = snbt.split('[')[0]  # Extract base name from SNBT
        if base_name not in base_name_to_new_id:
            base_name_to_new_id[base_name] = new_id
            new_id_to_base_name[new_id] = base_name
            base_name_to_full_snbt[base_name] = snbt  # Store the first full SNBT for this base name
            new_id += 1
        old_to_new_id[old_id] = base_name_to_new_id[base_name]

    # Process the npy file and count total blocks for each new ID
    block_ids = np.load(npy_path)
    new_block_ids = np.vectorize(old_to_new_id.get)(block_ids)
    unique, counts = np.unique(new_block_ids, return_counts=True)
    id_counts = {int(k): int(v) for k, v in zip(unique, counts)}  # Ensure keys and values are standard Python types

    # Create the new id_to_snbt mapping
    new_id_to_snbt = {new_id: base_name_to_full_snbt[base_name] for new_id, base_name in new_id_to_base_name.items()}

    # Save the new mapping
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'id_to_snbt.json'), 'w') as f:
        json.dump(new_id_to_snbt, f, indent=2)

    # Save id counts
    with open(os.path.join(output_path, 'id_counts.json'), 'w') as f:
        json.dump(id_counts, f, indent=2)

    # Save old-to-new ID mapping
    with open(os.path.join(output_path, 'id_to_id.json'), 'w') as f:
        json.dump(old_to_new_id, f, indent=2)

    # Save the new npy file
    np.save(os.path.join(output_path, 'new_block_ids.npy'), new_block_ids)

    # Save meta.json
    meta_data = {
        "original_shape": block_ids.shape,
        "new_shape": new_block_ids.shape,
        "num_unique_ids": len(new_id_to_snbt),
    }
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"Stage 1 completed. Mapped {len(snbt_to_id)} ids to {len(new_id_to_snbt)} new ids.")

def stage_2(folder_name, output_base_path):
    """
    Stage 2: Filter out blocks appearing with probability less than 0.00005 and merge them into a 'null' block.
    Regenerate id_to_id, id_to_snbt, and meta files based on the filtered data.
    """
    input_path = os.path.join(output_base_path, folder_name, 's1', 'id_counts.json')
    npy_path = os.path.join(output_base_path, folder_name, 's1', 'new_block_ids.npy')
    s1_id_to_snbt_path = os.path.join(output_base_path, folder_name, 's1', 'id_to_snbt.json')
    output_path = os.path.join(output_base_path, folder_name, 's2')

    with open(input_path, 'r') as f:
        id_counts = json.load(f)

    with open(s1_id_to_snbt_path, 'r') as f:
        s1_id_to_snbt = json.load(f)

    total_blocks = sum(id_counts.values())
    threshold = 0.00005 * total_blocks

    # Identify IDs to merge into 'null'
    null_id = max(map(int, id_counts.keys())) + 1
    filtered_id_counts = {}
    id_to_null = {}

    for id_str, count in id_counts.items():
        id_int = int(id_str)
        if count < threshold:
            id_to_null[id_int] = null_id
        else:
            filtered_id_counts[id_int] = count

    # Update the block IDs in the npy file
    block_ids = np.load(npy_path)
    new_block_ids = np.vectorize(lambda x: id_to_null.get(x, x))(block_ids)
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'filtered_block_ids.npy'), new_block_ids)

    # Update id_counts with the 'null' block
    filtered_id_counts[null_id] = sum(id_counts[str(id)] for id in id_to_null)

    # Save the updated id_counts
    with open(os.path.join(output_path, 'filtered_id_counts.json'), 'w') as f:
        json.dump(filtered_id_counts, f, indent=2)

    # Regenerate id_to_id and id_to_snbt mappings
    id_to_id = {}
    new_id_to_snbt = {}
    for old_id, new_id in id_to_null.items():
        id_to_id[old_id] = new_id
    for id_int in filtered_id_counts.keys():
        if id_int == null_id:
            new_id_to_snbt[id_int] = "null"
        else:
            new_id_to_snbt[id_int] = s1_id_to_snbt[str(id_int)]

    with open(os.path.join(output_path, 'id_to_snbt.json'), 'w') as f:
        json.dump(new_id_to_snbt, f, indent=2)

    with open(os.path.join(output_path, 'id_to_id.json'), 'w') as f:
        json.dump(id_to_id, f, indent=2)

    # Save meta.json
    meta_data = {
        "original_shape": block_ids.shape,
        "new_shape": new_block_ids.shape,
        "num_unique_ids": len(filtered_id_counts),
    }
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"Stage 2 completed. Filtered out blocks with probability < 0.00005 and merged them into 'null' block.")

def stage_3(folder_name, output_base_path):
    """
    Stage 3: Map each ID to a character in ASCII and generate the required files.
    """
    input_path = os.path.join(output_base_path, folder_name, 's2', 'filtered_id_counts.json')
    npy_path = os.path.join(output_base_path, folder_name, 's2', 'filtered_block_ids.npy')
    s2_id_to_snbt_path = os.path.join(output_base_path, folder_name, 's2', 'id_to_snbt.json')
    output_path = os.path.join(output_base_path, folder_name, 's3')

    with open(input_path, 'r') as f:
        id_counts = json.load(f)

    with open(s2_id_to_snbt_path, 'r') as f:
        s2_id_to_snbt = json.load(f)

    # Map IDs to ASCII characters
    ascii_chars = list(string.ascii_lowercase + string.ascii_uppercase)
    id_to_char = {}
    char_to_snbt = {}
    for i, id_int in enumerate(sorted(id_counts.keys())):
        char = ascii_chars[i % len(ascii_chars)]
        id_to_char[int(id_int)] = char
        char_to_snbt[char] = s2_id_to_snbt[str(id_int)]

    # Update the block IDs in the npy file
    block_ids = np.load(npy_path)
    char_block_ids = np.vectorize(id_to_char.get)(block_ids)
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'char_block_ids.npy'), char_block_ids)

    # Save id_to_char mapping
    with open(os.path.join(output_path, 'id_to_char.json'), 'w') as f:
        json.dump(id_to_char, f, indent=2)

    # Save char_to_snbt mapping
    with open(os.path.join(output_path, 'char_to_snbt.json'), 'w') as f:
        json.dump(char_to_snbt, f, indent=2)

    # Save meta.json
    meta_data = {
        "original_shape": block_ids.shape,
        "new_shape": char_block_ids.shape,
        "num_unique_chars": len(id_to_char),
    }
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"Stage 3 completed. Mapped IDs to ASCII characters and generated required files.")

def npy_to_jsonl(folder_name, output_base_path):
    """
    Transform the dataset of npy files into a set of strings stored in JSONL format.
    Each line represents a block of 16x16x16, formatted with spaces between characters,
    '#' at the end of each 16-character line, '$' at the end of each 16x16 block, and '%' at the end.
    """
    input_path = os.path.join(output_base_path, folder_name, 's3', 'char_block_ids.npy')
    output_path = os.path.join(output_base_path, folder_name, 's3', 'dataset.jsonl')

    # Load the npy file
    block_ids = np.load(input_path)

    # Ensure the inner dimensions are 16x16x16
    assert block_ids.shape[1] == 16
    assert block_ids.shape[2] == 16
    assert block_ids.shape[3] == 16

    # Process the data into JSONL format
    with open(output_path, 'w') as f:
        for data_block in block_ids:
            lines = []
            for layer in data_block:
                for row in layer:
                    lines.append(' '.join(row) + ' #')
                lines.append('$')
            lines.append('%')
            f.write(json.dumps({'data': ''.join(lines)}) + '\n')

    print(f"Dataset transformed to JSONL format and saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser(description="Post-process Minecraft block data.")
    parser.add_argument("-s", "--stage", type=int, default=None, help="Stage number to execute (default: run all stages).")
    parser.add_argument("--folder-name", type=str, required=True, help="Name of the folder containing input data.")
    args = parser.parse_args()

    output_base_path = "data_s1"  # Base output path for all stages

    if args.stage is None or args.stage == 1:
        stage_1(args.folder_name, output_base_path)
    if args.stage is None or args.stage == 2:
        stage_2(args.folder_name, output_base_path)
    if args.stage is None or args.stage == 3:
        stage_3(args.folder_name, output_base_path)
    if args.stage is None or args.stage == 4:
        npy_to_jsonl(args.folder_name, output_base_path)

if __name__ == "__main__":
    main()
