"""
================================================================================
SCIENCE FOUNDATION DATA EXTRACTOR: PHYSICS, CHEMISTRY, & LIFE SCIENCES
================================================================================
DATASETS TARGETED:
- [opendatalab/Sci-Base](https://huggingface.co/datasets/opendatalab/Sci-Base) (Configurations: 'paper', 'textbook')

This dataset was released on huggingface on March 27, 2026
As Mistral Small 4 (119B-parameter Mixture-of-Experts model) was released on March 16, 2026
and Ministral 3 14B was released on December 2, 2025, this is a good pretraining dataset
to test their knowledge accretion.

OVERVIEW:
This script extracts a ~1B token "Science Foundation" corpus. By targeting 
Physics, Chemistry, and Life Sciences, we capture the cross-disciplinary 
data necessary for advanced scientific reasoning and simulation-agent tasks.

HOW IT WORKS:
1. STREAMING: Uses 'datasets' streaming to process the 3.9TB repository shard 
   by shard without requiring massive local storage.
2. DISCIPLINE FILTERING: Inspects 'sci_category'. We specifically include 
   'life sciences' alongside 'physics' and 'chemistry' to capture the 
   broadest range of hard-science data.
3. TEXT RECONSTRUCTION: Flattens the MinerU-parsed 'content_list' into clean 
   text strings, strictly filtering out null/None values.
4. LUSTRE OPTIMIZATION: Saves data in Parquet shards to ensure high-speed 
   I/O and parallel loading for Axolotl during training.

DEPENDENCIES:
- pip install datasets pandas pyarrow tqdm huggingface_hub
================================================================================
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Configuration
SAVE_DIR = "./sci-base-science-local"
TARGET_CATEGORIES = ["physics", "chemistry", "life sciences"]
CONFIGS = ['paper', 'textbook']
# XXX JB optimize value for contiguous training data of larger size more suited to lustre performance
#SHARD_SIZE = 2500
SHARD_SIZE = 20000

os.makedirs(SAVE_DIR, exist_ok=True)

def extract_text(content_list):
    """Flattens content_list, strictly ignoring NoneTypes and non-strings."""
    if not content_list or not isinstance(content_list, list):
        return ""
    
    texts = []
    for item in content_list:
        if isinstance(item, dict):
            val = item.get('text')
            if isinstance(val, str) and val.strip():
                texts.append(val)
    
    return "\n".join(texts)

def stream_and_filter():
    buffer = []
    total_saved = 0
    shard_count = 0

    for config in CONFIGS:
        print(f"\n--- Starting Stream: opendatalab/Sci-Base [{config}] ---")
        
        # Load the stream from Hugging Face
        ds = load_dataset('opendatalab/Sci-Base', config, streaming=True, split='train')
        ds_iter = iter(ds)
        
        while True:
            try:
                entry = next(ds_iter)
            except StopIteration:
                break
            except Exception:
                continue 

            # Filtering logic
            category = str(entry.get('sci_category', '')).lower()
            
            if any(target in category for target in TARGET_CATEGORIES):
                raw_text = extract_text(entry.get("content_list", []))
                
                if raw_text.strip():
                    buffer.append({
                        "title": entry.get("title", ""),
                        "category": category,
                        "text": raw_text
                    })
            
            # Save shard to Lustre
            if len(buffer) >= SHARD_SIZE:
                shard_path = os.path.join(SAVE_DIR, f"science_shard_{shard_count}.parquet")
                pd.DataFrame(buffer).to_parquet(shard_path, engine='pyarrow', index=False)
                
                total_saved += len(buffer)
                shard_count += 1
                buffer = []
                print(f"| Shards: {shard_count} | Total Rows: {total_saved} | Domain: {category}", end="\r")

    # Final flush
    if buffer:
        shard_path = os.path.join(SAVE_DIR, f"science_shard_{shard_count}.parquet")
        pd.DataFrame(buffer).to_parquet(shard_path, engine='pyarrow', index=False)
        total_saved += len(buffer)

    print(f"\n\nSuccess! Science Foundation (1B Token Goal) saved to: {SAVE_DIR}")

if __name__ == "__main__":
    stream_and_filter()

