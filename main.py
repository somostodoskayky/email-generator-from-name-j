import os
import re
import time
import numpy as np
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# =============== SMART BATCH PROCESSING ==================
# =========================================================
alpha_re = re.compile(r'[^A-Za-z]+')

def normalize_safe(name: str) -> str:
    """Keep only alphabetic characters."""
    if not name:
        return ""
    cleaned = alpha_re.sub("", str(name))
    return cleaned

def process_names_batch_smart(first_names, last_names):
    # First names: take first 3-5 valid letters
    first_cleaned = [normalize_safe(f).lower() for f in first_names]
    first_cleaned = [f[:5] if len(f) >= 3 else (f + "xxx")[:3] for f in first_cleaned]

    # Last names: at least 5 letters, pad if shorter
    last_cleaned = [normalize_safe(l).lower() for l in last_names]
    last_cleaned = [l if len(l) >= 5 else (l + "xxxxx")[:5] for l in last_cleaned]

    return first_cleaned, last_cleaned

# ---------------- Worker-side generator ----------------
def _generate_chunk(first_chunk, last_chunk, domain, seed):
    rng = np.random.default_rng(seed)
    n = len(first_chunk)

    # Ensure all elements are strings
    first_arr = np.array([str(f) for f in first_chunk], dtype=object)
    last_arr  = np.array([str(l) for l in last_chunk], dtype=object)

    # Random numbers at the end (1-4 digits)
    numbers = rng.integers(1, 10000, n).astype(str)

    # Lowercase everything safely
    fn_arr = np.array([f.lower() for f in first_arr], dtype=object)
    ln_arr = np.array([l.lower() for l in last_arr], dtype=object)

    def safe_concat(*parts):
        out = ""
        for part in parts:
            if not part:
                continue
            if out and out[-1] in "._" and part[0] in "._":
                out += part[1:]
            else:
                out += part
        return out

    # Patterns: only '.' or '_', 1 or 2 separators
    patterns = [
        lambda fn, ln, num: safe_concat(fn, ".", ln, num),
        lambda fn, ln, num: safe_concat(ln, ".", fn, num),
        lambda fn, ln, num: safe_concat(fn, "_", ln, num),
        lambda fn, ln, num: safe_concat(ln, "_", fn, num),
    ]
    pattern_indices = rng.integers(0, len(patterns), n)
    local_parts = np.empty(n, dtype=object)
    for i in range(n):
        local_parts[i] = patterns[pattern_indices[i]](fn_arr[i], ln_arr[i], numbers[i])

    return np.char.add(local_parts, f"@{domain}")

def generate_emails_smart_batch_parallel(first_names, last_names, domain, workers=None, chunk_size=200_000, base_seed=12345):
    n = len(first_names)
    if n == 0:
        return np.array([], dtype=object)

    if workers is None:
        workers = max(1, min(os.cpu_count() or 1, 16))

    chunks = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]
    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_generate_chunk, first_names[s:e], last_names[s:e], domain, base_seed + idx*100_003)
                   for idx, (s, e) in enumerate(chunks)]
        for i, fut in enumerate(as_completed(futures)):
            results[i] = fut.result()
    return np.concatenate(results)

# =========================================================
# ================= MAIN CSV PROCESSOR ====================
# =========================================================
def process_csv(input_file: str,
                output_file: str = "generated_emails.csv",
                domain: str = "example.com",
                workers: int | None = None,
                chunk_size: int = 200_000) -> None:
    start_time = time.time()
    if os.path.exists(output_file):
        os.remove(output_file)

    print("🚀 POLARS + multiprocess fast path")

    df_pl = pl.read_csv(input_file, columns=["first", "last"], infer_schema_length=0)
    df_pl = df_pl.with_columns([
        pl.col("first").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("last").cast(pl.Utf8, strict=False).fill_null("")
    ])

    first_list = df_pl["first"].to_list()
    last_list  = df_pl["last"].to_list()
    first_clean, last_clean = process_names_batch_smart(first_list, last_list)

    emails = generate_emails_smart_batch_parallel(first_clean, last_clean, domain, workers=workers, chunk_size=chunk_size)
    df_pl = df_pl.with_columns(pl.Series("email", emails))
    df_pl.write_csv(output_file)

    elapsed_time = time.time() - start_time
    print(f"\n✅ Emails generated and saved to {output_file}")
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")

# =========================================================
# ======================== CLI ============================
# =========================================================
if __name__ == "__main__":
    process_csv("names.csv", "generated_emails.csv", domain="example.com", workers=None, chunk_size=200_000)
