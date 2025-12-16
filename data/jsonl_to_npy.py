import os
import json
import numpy as np

def jsonl_to_npy(input_jsonl_path, folder_name, output_base_path, line_index_arg):
    """
    Parse a JSONL of strings back into an npy array of indexes and map IDs to blocks.
    Directly use char_to_snbt from Stage 3 for back mapping to IDs.
    Ensure parsing correctly handles concatenated strings and ignores delimiters.
    """
    char_to_snbt_path = os.path.join(output_base_path, folder_name, 's3', 'char_to_snbt.json')
    output_npy_path = os.path.join(output_base_path, folder_name, 'sf', 'parsed_block_ids.npy')

    # Load mappings
    with open(char_to_snbt_path, 'r') as f:
        char_to_snbt = json.load(f)

    # Generate a new id_to_snbt mapping
    id_to_snbt = {str(i): snbt for i, snbt in enumerate(char_to_snbt.values())}
    
    # Save the new id_to_snbt mapping in the same folder structure as s1, s2, s3
    sf_folder_path = os.path.join(output_base_path, folder_name, 'sf')
    os.makedirs(sf_folder_path, exist_ok=True)
    output_id_to_snbt_path = os.path.join(sf_folder_path, 'id_to_snbt.json')
    
    # Reverse mappings
    snbt_to_id = {v: int(k) for k, v in id_to_snbt.items()}

    # Ensure null blocks map to bedrock before saving
    id_to_snbt[str(len(id_to_snbt))] = "universal_minecraft:bedrock"
    snbt_to_id["null"] = snbt_to_id.get("universal_minecraft:bedrock", len(snbt_to_id))

    # Save the new id_to_snbt mapping
    with open(output_id_to_snbt_path, 'w') as f:
        json.dump(id_to_snbt, f, indent=4)

    # Parse JSONL and reconstruct npy array
    blocks = []
    with open(input_jsonl_path, 'r') as f:
        lines = f.readlines()

    # Use the provided line index argument
    line_index = line_index_arg
    if line_index < 0 or line_index >= len(lines):
        raise IndexError(f"Line index {line_index} is out of range. Total lines: {len(lines)}")

    # Process only the selected line
    data = json.loads(lines[line_index])['data']
    data = data.replace('%', '')  # Remove '%' delimiters
    block = []
    layers = data.split('$')[:-1]  # Split by '$', remove the last empty split
    for layer in layers:
        rows = layer.split('#')[:-1]  # Split by '#', remove the last empty split
        block_layer = []
        for row in rows:
            chars = row.split(' ')[:-1]  # Split by '#', remove the last empty split
            block_layer_row = []
            block_layer_row.extend([snbt_to_id[char_to_snbt[char]] for char in chars])
            assert len(block_layer_row) == 16, f"Each row must have 16 characters. Found: {len(block_layer_row)}"
            block_layer.append(block_layer_row)
        assert len(block_layer) == 16, f"Each layer must have 16 rows. Found: {len(block_layer)}"
        block.append(block_layer)
    assert len(block) == 16, f"Each block must have 16 layers. Found: {len(block)}"
    blocks.append(block)

    # Convert to numpy array
    block_ids = np.array(block, dtype=np.int32)  # Save only the block, not as a nested array
    assert block_ids.shape == (16, 16, 16), f"Output NPY shape must be (16, 16, 16). Found: {block_ids.shape}"
    np.save(output_npy_path, block_ids)

    print(f"Line {line_index} parsed back to npy array and saved to {output_npy_path}.")
    print(f"New id_to_snbt mapping saved to {output_id_to_snbt_path}.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse JSONL back to NPY array.")
    parser.add_argument("--input-jsonl-path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--folder-name", type=str, required=True, help="Name of the folder containing input data.")
    parser.add_argument("-l", "--line-index", type=int, required=True, help="Line index to process (0-based).")
    args = parser.parse_args()

    output_base_path = "data_s1"  # Base output path for all stages
    jsonl_to_npy(args.input_jsonl_path, args.folder_name, output_base_path, args.line_index)