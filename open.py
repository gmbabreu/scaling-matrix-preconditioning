import os
import hashlib
import numpy as np
import pickle
from array import array
from tqdm import tqdm
from datasets import load_dataset

allowed_tokens = [
    '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'
]
stoi = {token: i for i, token in enumerate(allowed_tokens)}
itos = {i: token for i, token in enumerate(allowed_tokens)}

VAL_FRAC = 0.0005
TRAIN_FLUSH_EVERY = 10_000_000
VAL_FLUSH_EVERY  = 1_000_000

def encode_or_none(text):
    """Encode text to list of token ids. Returns None if any char is OOV."""
    out = array('B')
    for char in text:
        idx = stoi.get(char)
        if idx is None:
            return None
        out.append(idx)
    return out

def is_val(text):
    h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    return (h % 10000) < int(VAL_FRAC * 10000)

if __name__ == '__main__':
    os.makedirs('open', exist_ok=True)
    for f in ['open/train.bin', 'open/val.bin']:
        if os.path.exists(f): os.remove(f)

    dataset = load_dataset("openwebtext", split="train", streaming=True)
    dataset = dataset.shuffle(seed=2357, buffer_size=10_000)

    train_buf = array('B')
    val_buf   = array('B')
    n_docs = 0
    n_filtered = 0

    print("Streaming and processing dataset...")
    for example in tqdm(dataset, desc="processing", total=8_013_769):
        text = example['text']

        ids = encode_or_none(text)
        if ids is None:
            n_filtered += 1
            continue

        ids.append(stoi['\n'])
        n_docs += 1

        if is_val(text):
            val_buf.extend(ids)
            if len(val_buf) >= VAL_FLUSH_EVERY:
                with open('open/val.bin', 'ab') as f:
                    np.frombuffer(val_buf, dtype=np.uint8).tofile(f)
                val_buf = array('B')
        else:
            train_buf.extend(ids)
            if len(train_buf) >= TRAIN_FLUSH_EVERY:
                with open('open/train.bin', 'ab') as f:
                    np.frombuffer(train_buf, dtype=np.uint8).tofile(f)
                train_buf = array('B')

    # Final flush
    with open('open/train.bin', 'ab') as f:
        np.frombuffer(train_buf, dtype=np.uint8).tofile(f)
    with open('open/val.bin', 'ab') as f:
        np.frombuffer(val_buf, dtype=np.uint8).tofile(f)

    meta = {'vocab_size': len(allowed_tokens), 'itos': itos, 'stoi': stoi}
    with open('open/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    total = n_docs + n_filtered
    print(f"Done. Kept {n_docs}/{total} docs ({100*n_docs/max(total,1):.1f}%) after filtering.")
    print(f"train.bin: {os.path.getsize('open/train.bin') / 1e9:.2f} GB")
    print(f"val.bin:   {os.path.getsize('open/val.bin') / 1e6:.2f} MB")
