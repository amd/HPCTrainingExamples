
import torch
import numpy as np
from transformers import AutoTokenizer
import os

class SFTTokenizerWrapper:
    def __init__(self, local_model_path, max_length=256):
        print(f"Loading tokenizer from: {local_model_path}")

        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local path not found: {local_model_path}")

        # load local model
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.inst_end_token = "[/INST]"

    def validate_batch(self, texts):
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        results = []

        for i in range(len(texts)):
            full_text = texts[i]
            if self.inst_end_token in full_text:
                parts = full_text.split(self.inst_end_token)
                prompt_text = parts[0] + self.inst_end_token

                # encode prompt with BOS handling
                prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
                mask_limit = min(len(prompt_token_ids), self.max_length)

                # apply the -100 mask to labels
                labels[i, :mask_limit] = -100

            # mask padding
            labels[i][input_ids[i] == self.tokenizer.pad_token_id] = -100

            # extraction for report
            unmasked_indices = np.where(labels[i] != -100)[0]
            if len(unmasked_indices) > 0:
                first_idx = unmasked_indices[0]
                first_token_text = self.tokenizer.decode([input_ids[i][first_idx]])
                last_masked_token = self.tokenizer.decode([input_ids[i][first_idx - 1]]) if first_idx > 0 else "N/A"
            else:
                first_idx, first_token_text, last_masked_token = None, "NONE", "NONE"

            results.append({
                "prompt_sample": texts[i][:60],
                "first_unmasked_idx": first_idx,
                "last_masked_token": last_masked_token,
                "first_unmasked_token": first_token_text
            })

        return results


if __name__ == "__main__":

    model = "/p/lustre5/belof1/hfmodels/Ministral-3-14B-Base-2512"

    test_data = [
        "[INST] What is the HBM capacity of an MI300A? [/INST] Each MI300A APU has 128GB of HBM3 memory.",
        "[INST] Write a hello world in Python. [/INST] print('Hello, World!')",
    ]

    try:
        validator = SFTTokenizerWrapper(model)
        outputs = validator.validate_batch(test_data)

        print("-" * 50)
        print("MASKING VALIDATION REPORT")
        print("-" * 50)

        for res in outputs:
            print(f"Sample: {res['prompt_sample']}...")
            print(f"  Last Masked Token (Should be [/INST]): {res['last_masked_token']}")
            print(f"  First Learned Token (Should be Response): {res['first_unmasked_token']}")
            print(f"  Boundary Index: {res['first_unmasked_idx']}")
            print("-" * 50)

            if "[/INST]" not in res['last_masked_token']:
                print("WARNING: The boundary token was not detected correctly!")

    except Exception as e:
        print(f"Error during validation: {e}")


