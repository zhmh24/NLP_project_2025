import amulet
import argparse
import logging
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
from pathlib import Path
from collections import Counter
import math

random.seed(42)

arg_parser = argparse.ArgumentParser(description="Process Minecraft world data to extract 16x16x16 block ID samples near the surface.")
arg_parser.add_argument("--folder", type=str, required=True, help="Minecraft server folder name under worlds/")
arg_parser.add_argument("--chunk-radius", type=int, default=64, help="Chunk radius around (0,0)")
arg_parser.add_argument("--region-chunk-radius", type=int, default=1, help="Side length of each region in chunks")
arg_parser.add_argument("--region-height", type=int, default=16, help="Height of each region")
arg_parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4, help=f"Number of worker threads (default: {os.cpu_count() or 4})")
arg_parser.add_argument("-s", "--sample-chunks", type=int, default=-1, help="Number of chunks to randomly sample")
args = arg_parser.parse_args()

SERVER_DIR = Path(f"worlds/{args.folder}")
OUTPUT_DIR = Path(f"data_s2/{args.folder}")
REGION_CHUNK_RADIUS = args.region_chunk_radius
REGION_HEIGHT = args.region_height

def get_biome_at(chunk, x, y, z):
    """
    For Minecraft 1.21.1: Biomes are stored in 4x4x4 cells.
    We must divide block coordinates by 4 to get the correct index.
    """
    try:
        # bx, bz will be between 0-3 for a standard chunk
        bx = (x % 16) >> 2
        bz = (z % 16) >> 2
        
        # by depends on the height (e.g., Y=64 -> by=16)
        by = y >> 2
        
        # Amulet stores biomes in a 3D numpy array [x, y, z]
        biome_index = chunk.biomes[bx, by, bz]
        return chunk.biome_palette[biome_index]
    except Exception as e:
        logging.debug(f"Biome lookup failed: {e}")
        return "minecraft:plains"

def collect_chunk_metadata_and_snbt(world_obj, cx, cz):
    min_y_dim, max_y_dim = 32, 256
    local_snbt_counts = Counter()
    base_y = -1
    surface_ys = []
    std_ys = -1.0
    chunk_biome = "unknown"

    try:
        # We'll use the center chunk of the region to sample the biome
        center_chunk = world_obj.get_chunk(cx, cz, "minecraft:overworld")
        
        for dcx in range(cx, cx + REGION_CHUNK_RADIUS):
            for dcz in range(cz, cz + REGION_CHUNK_RADIUS):
                chunk = world_obj.get_chunk(dcx, dcz, "minecraft:overworld")
                
                # Optimization: Use numpy to find surface height faster
                # block_array_cy = chunk.blocks[:, min_y_dim:max_y_dim, :]
                
                # for x in [4, 12]:
                #     for z in [4, 12]:
                #         # Check for water at one of the sample points
                #         if x == 4 and z == 4:
                #             water_block = chunk.get_block(x, block_array_cy.shape[1] + min_y_dim -1, z)
                #             if 'water' in water_block.base_name:
                #                 return {'cx': cx, 'cz': cz, 'snbt_counts': Counter(), 'base_y': -1, 'std_ys': -1.0, 'biome': "ocean"}

                #         non_air_indices = np.nonzero(np.atleast_1d(block_array_cy[x, :, z] != 0))[0]
                #         if non_air_indices.size > 0:
                #             surface_ys.append(non_air_indices[-1] + min_y_dim)
                for y_coord in range(max_y_dim - 1, min_y_dim - 1, -1):
                    block = chunk.get_block(8, y_coord, 8)
                    if block.base_name == "water":
                        return cx, cz, Counter(), -1, -1.0, "ocean"
                    elif block.base_name != "air":
                        surface_ys.append(y_coord)
                        break
        if not surface_ys or min_y_dim in surface_ys:
            return cx, cz, Counter(), -1, -1.0, "unknown"
            
        mean_ys = sum(surface_ys) / len(surface_ys)
        std_ys = math.sqrt(sum((y - mean_ys) ** 2 for y in surface_ys) / len(surface_ys))
        base_y = int(mean_ys + random.uniform(-std_ys / 4, std_ys / 4) - random.uniform(0.3, 0.5) * REGION_HEIGHT)

        for x in [4, 12]:
            for z in [4, 12]:
                for dy in range(REGION_HEIGHT):
                    current_world_y = base_y + dy
                    block = center_chunk.get_block(x, current_world_y, z)
                    if block.base_name == "air":
                        surface_ys.append(current_world_y - 1)
                        break
                    if dy == REGION_HEIGHT - 1:
                        surface_ys.append(current_world_y)
        
        mean_ys = sum(surface_ys) / len(surface_ys)
        std_ys = math.sqrt(sum((y - mean_ys) ** 2 for y in surface_ys) / len(surface_ys))
        # Capture the biome at the surface level in the center of the chunk
        chunk_biome = get_biome_at(center_chunk, 8, int(mean_ys), 8)

        for dcx in range(cx, cx + REGION_CHUNK_RADIUS):
            for dcz in range(cz, cz + REGION_CHUNK_RADIUS):
                chunk = world_obj.get_chunk(dcx, dcz, "minecraft:overworld")
                for dx_in_chunk in range(16):
                    for dz_in_chunk in range(16):
                        for dy_offset in range(REGION_HEIGHT):
                            current_world_y = base_y + dy_offset
                            block = chunk.get_block(dx_in_chunk, current_world_y, dz_in_chunk)
                            snbt_string = block.snbt_blockstate
                            local_snbt_counts[snbt_string] += 1
                                    
    except amulet.api.errors.ChunkDoesNotExist:
        logging.warning(f"Chunk ({cx},{cz}) does not exist.")
    except Exception as e:
        logging.error(f"Error in Phase 1 for chunk ({cx},{cz}): {e}")
    return cx, cz, local_snbt_counts, base_y, std_ys, chunk_biome

def extract_block_id_array(world_obj, cx, cz, base_y, 
                           snbt_to_id, air_snbt_id):
    block_id_array = np.full((16 * REGION_CHUNK_RADIUS,
                              REGION_HEIGHT,
                              16 * REGION_CHUNK_RADIUS), air_snbt_id, dtype=np.uint64)

    try:
        for dcx in range(cx, cx + REGION_CHUNK_RADIUS):
            for dcz in range(cz, cz + REGION_CHUNK_RADIUS):
                chunk = world_obj.get_chunk(dcx, dcz, "minecraft:overworld")
                for dx_in_chunk in range(16):
                    for dz_in_chunk in range(16):
                        for dy_offset in range(REGION_HEIGHT):
                            current_world_y = base_y + dy_offset
                            block = chunk.get_block(dx_in_chunk, current_world_y, dz_in_chunk)
                            snbt_string = block.snbt_blockstate
                            block_id_array[dx_in_chunk + (dcx - cx) * 16,
                                           dy_offset,
                                           dz_in_chunk + (dcz - cz) * 16] = snbt_to_id.get(snbt_string, air_snbt_id)
    
    except amulet.api.errors.ChunkDoesNotExist:
        logging.warning(f"Chunk ({cx},{cz}) vanished before ID array extraction. Resulting array is air-filled.")
    except Exception as e:
        logging.error(f"Error extracting block ID array for chunk ({cx},{cz}): {e}", exc_info=True)
                
    return block_id_array

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"Loading world from {SERVER_DIR}...")
    try:
        world = amulet.load_level(str(SERVER_DIR))
    except Exception as e:
        logging.error(f"Failed to load world: {e}")
        return
    
    target_chunks_coords = [(cx, cz) for cx in range(-args.chunk_radius, args.chunk_radius - REGION_CHUNK_RADIUS, REGION_CHUNK_RADIUS)
                            for cz in range(-args.chunk_radius, args.chunk_radius - REGION_CHUNK_RADIUS, REGION_CHUNK_RADIUS)]

    if args.sample_chunks != -1:
        if args.sample_chunks > len(target_chunks_coords):
            logging.warning(f"Requested sample size ({args.sample_chunks}) exceeds total chunks ({len(target_chunks_coords)}). Processing all chunks.")
        else:
            # Generate a mask to randomly select chunks while preserving order
            mask = [1 if i < args.sample_chunks else 0 for i in range(len(target_chunks_coords))]
            random.shuffle(mask)  # Shuffle the mask to randomly select chunks
            target_chunks_coords = [chunk for chunk, m in zip(target_chunks_coords, mask) if m == 1]
    
    all_snbt_counts = Counter()
    chunk_processing_params = []
    
    # NEW: List to store dictionaries of chunk metadata
    chunk_metadata = []

    logging.info("Phase 1: Collecting SNBT, Surface Y, and Biomes...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(collect_chunk_metadata_and_snbt, world, cx, cz)
                   for cx, cz in target_chunks_coords]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1"):
            try:
                r_cx, r_cz, local_snbt_counts, base_y, std_ys, chunk_biome = future.result()
                if base_y < 0:
                    continue
                all_snbt_counts.update(local_snbt_counts)
                chunk_metadata.append({'cx': r_cx, 'cz': r_cz, 'snbt_counts': local_snbt_counts, 'base_y': base_y, 'std_ys': std_ys, 'biome': chunk_biome})
            except Exception as e:
                logging.error(f"Error in Phase 1 future: {e}")
    
    if not chunk_metadata:
        logging.info("No valid regions found. Exiting.")
        world.close()
        return

    # Sort the collected metadata by chunk coordinates to ensure consistent order
    chunk_metadata.sort(key=lambda p: (p['cx'], p['cz']))

    # Now that it's sorted, we can create the params for the next phase
    chunk_processing_params = [{'cx': p['cx'], 'cz': p['cz'], 'base_y': p['base_y']} for p in chunk_metadata]
    stats_and_biomes = [(p['std_ys'], p['biome']) for p in chunk_metadata]

    # Palette generation (unchanged)
    snbt_counts_path = OUTPUT_DIR / "snbt_counts.json"
    with open(snbt_counts_path, 'w') as f:
        json.dump(all_snbt_counts, f, indent=2, sort_keys=True)

    air_snbt_str = amulet.Block("universal_minecraft", "air").snbt_blockstate
    unique_snbt_for_palette = set(all_snbt_counts.keys())
    unique_snbt_for_palette.discard(air_snbt_str) 
    sorted_other_snbt = sorted(list(unique_snbt_for_palette))
    
    snbt_to_id = {air_snbt_str: 0}
    snbt_to_id.update({snbt: i + 1 for i, snbt in enumerate(sorted_other_snbt)})
    air_snbt_id_val = 0 

    id_to_snbt = {i: snbt for snbt, i in snbt_to_id.items()}
    palette_map_path = OUTPUT_DIR / "id_to_snbt.json" # Corrected path
    with open(palette_map_path, 'w') as f:
        json.dump(id_to_snbt, f, indent=2, sort_keys=True)
    logging.info(f"Saved SNBT palette map to {palette_map_path}")

        # Phase 2: Extraction
    all_block_id_arrays = []
    logging.info(f"Phase 2: Extracting {len(chunk_processing_params)} block ID arrays...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit tasks and keep futures in order
        futures = [executor.submit(extract_block_id_array, world, p['cx'], p['cz'], p['base_y'], snbt_to_id, air_snbt_id_val) for p in chunk_processing_params]
        
        # Retrieve results in the same order they were submitted
        for future in tqdm(futures, total=len(futures), desc="Phase 2"):
            try:
                block_id_array = future.result()
                all_block_id_arrays.append(block_id_array)
            except Exception as e:
                logging.error(f"Critical error retrieving block ID array from Phase 2 task: {e}", exc_info=True)

    world.close()

    # SAVE FINAL DATA
    if all_block_id_arrays:
        # 1. Save Block IDs
        np.save(OUTPUT_DIR / "block_ids.npy", np.stack(all_block_id_arrays, axis=0))
        
        # 2. Save Stats and Biomes using a structured array
        # f4 = 32-bit float, U64 = Unicode string (max 64 chars)
        dtype = [('std_dev', 'f4'), ('biome', 'U64')]
        metadata_stats_array = np.array(stats_and_biomes, dtype=dtype)
        
        output_stats_path = OUTPUT_DIR / "chunk_metadata_stats.npy"
        np.save(output_stats_path, metadata_stats_array)
        
        logging.info(f"Saved block IDs and metadata (biomes + std_ys) to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()