import amulet
import numpy as np
import json
from pathlib import Path
from amulet.api import Block
import argparse
import logging

logging.basicConfig(level=logging.INFO)

argparser = argparse.ArgumentParser(description="Visualize output in minecraft.")
argparser.add_argument("--cx", type=int, default=0, help="Base X coordinate of the chunk")
argparser.add_argument("--y", type=int, default=0, help="Base Y coordinate of the region")
argparser.add_argument("--cz", type=int, default=0, help="Base Z coordinate of the chunk")
argparser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
argparser.add_argument("--world_path", type=str, default="world", help="Path to the world")
args = argparser.parse_args()

DATA_PATH = Path(args.data_path)

region = np.load(DATA_PATH / "sf/parsed_block_ids.npy")
x_len, y_len, z_len = region.shape
cx_len = x_len // 16
cz_len = z_len // 16

id_to_snbt = {int(k): v for k, v in json.load(open(DATA_PATH / "sf/id_to_snbt.json")).items()}

level = amulet.load_level(args.world_path)

for cx in range(args.cx, args.cx + cx_len):
    for cz in range(args.cz, args.cz + cz_len):
        chunk = level.get_chunk(cx, cz, "minecraft:overworld")
        for x_offset in range(16):
            for y_offset in range(y_len):
                for z_offset in range(16):
                    block_id = int(region[(cx - args.cx) * 16 + x_offset, y_offset, (cz - args.cz) * 16 + z_offset])
                    snbt = id_to_snbt.get(block_id, id_to_snbt[0])  # Default to air if not found
                    chunk.set_block(x_offset, y_offset + args.y,
                                    z_offset, Block.from_snbt_blockstate(snbt))
        chunk.changed = True
logging.info("Finished setting blocks")
level.save()
level.close()
