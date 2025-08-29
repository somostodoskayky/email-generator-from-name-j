import os
import re
import time
import math
import numpy as np
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# =============== SMART BATCH PROCESSING ==================
# =========================================================
alpha_re = re.compile(r'[^A-Za-z]+')

def normalize_safe(name: str) -> str:
    if not name:
        return "user"
    cleaned = alpha_re.sub("", str(name))
    return cleaned or "user"

def process_names_batch_smart(first_names, last_names):
    return [normalize_safe(f) for f in first_names], [normalize_safe(l) for l in last_names]

# ---------------- Worker-side generator ----------------
def _generate_chunk(first_chunk, last_chunk, domain, seed):
    # Each process gets its own RNG for independence & speed
    rng = np.random.default_rng(seed)

    n = len(first_chunk)
    first_arr = np.asarray(first_chunk, dtype=object)
    last_arr  = np.asarray(last_chunk,  dtype=object)

    fallback_initials = np.array(['a', 'b', 'c', 'd'], dtype=object)

    # Precompute randomness
    # numbers are mandatory, 1-4 digits, always at the end
    numbers = rng.integers(1, 10000, n).astype(str)

    # Lowercase everything (enforce lowercase-only rule)
    casing = rng.choice([True, False], n * 6)
    pattern_indices = rng.integers(0, 75, n)
    letter_indices = rng.integers(0, 10, n * 4)  # reused in random pairs

    fi_indices = rng.integers(0, 4, n)
    li_indices = rng.integers(0, 4, n)

    # First & last initials (with fallback)
    fi_arr = np.empty(n, dtype=object)
    li_arr = np.empty(n, dtype=object)
    for i in range(n):
        f = first_arr[i]
        l = last_arr[i]
        fi_arr[i] = (f[0] if f else fallback_initials[fi_indices[i]])
        li_arr[i] = (l[0] if l else fallback_initials[li_indices[i]])

    # Random two-letter combos
    # Use the pre-drawn indices deterministically per row
    base = np.arange(n) * 4
    rfn_arr = np.empty(n, dtype=object)
    rln_arr = np.empty(n, dtype=object)
    for i in range(n):
        f = first_arr[i]
        l = last_arr[i]
        if len(f) >= 2:
            idx1 = letter_indices[base[i]]     % len(f)
            idx2 = letter_indices[base[i] + 1] % len(f)
            rfn_arr[i] = f[idx1] + f[idx2]
        else:
            rfn_arr[i] = f
        if len(l) >= 2:
            idx1 = letter_indices[base[i] + 2] % len(l)
            idx2 = letter_indices[base[i] + 3] % len(l)
            rln_arr[i] = l[idx1] + l[idx2]
        else:
            rln_arr[i] = l

    # Apply casing flags (lower if flag True)
    fn_arr  = np.array([str(s).lower() for s in first_arr], dtype=object)
    ln_arr  = np.array([str(s).lower() for s in last_arr], dtype=object)
    fi_arr  = np.array([str(s).lower() for s in fi_arr], dtype=object)
    li_arr  = np.array([str(s).lower() for s in li_arr], dtype=object)
    rfn_arr = np.array([str(s).lower() for s in rfn_arr], dtype=object)
    rln_arr = np.array([str(s).lower() for s in rln_arr], dtype=object)

    # Helper to safely concat, preventing double '.' or '_'
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

    # Patterns: only '.' and '_' as separators, 1 or 2 separators total, and number at end
    patterns = [
        # One separator (dot)
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ".", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ".", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, ".", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, ".", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", fi, num),
        # One separator (underscore)
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, "_", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, "_", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, "_", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", fi, num),
        # Two separators (dot)
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ".", li, ".", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", ln, ".", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, ".", fi, ".", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", fn, ".", fi, num),
        # Two separators (underscore)
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, "_", li, "_", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", ln, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, "_", fi, "_", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", fn, "_", fi, num),
        # One separator using random letters
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, ".", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", rln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", rln, num),
    ]

    # Compute pattern indices based on actual pattern count
    pattern_indices = rng.integers(0, len(patterns), n)

    # Build locals per pattern (single Python loop over rows)
    local_parts = np.empty(n, dtype=object)
    for i in range(n):
        if (first_arr[i] or last_arr[i]):
            p = pattern_indices[i]
            local_parts[i] = patterns[p](fn_arr[i], ln_arr[i], fi_arr[i], li_arr[i], rfn_arr[i], rln_arr[i], numbers[i])
        else:
            local_parts[i] = f"{fi_arr[i]}{li_arr[i]}{numbers[i]}"

    return np.char.add(local_parts, f"@{domain}")

def generate_emails_smart_batch_parallel(first_names, last_names, domain, workers=None, chunk_size=200_000, base_seed=12345):
    """Parallel version preserving original behavior."""
    n = len(first_names)
    if n == 0:
        return np.array([], dtype=object)

    if workers is None:
        workers = max(1, min(os.cpu_count() or 1, 16))  # cap a bit to avoid oversubscription

    # Create chunks
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((start, end))

    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for idx, (s, e) in enumerate(chunks):
            # Give each chunk a distinct seed (stable but independent)
            seed = base_seed + idx * 100_003
            futures.append(
                ex.submit(
                    _generate_chunk,
                    first_names[s:e],
                    last_names[s:e],
                    domain,
                    seed
                )
            )
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
    """High-performance CSV processing using Polars + multiprocessing."""
    start_time = time.time()

    if os.path.exists(output_file):
        os.remove(output_file)

    print("🚀 POLARS + multiprocess fast path (logic preserved)")

    # Read only required columns
    df_pl = pl.read_csv(input_file, columns=["first", "last"], infer_schema_length=0)

    # Clean columns
    df_pl = df_pl.with_columns([
        pl.col("first").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("last").cast(pl.Utf8, strict=False).fill_null("")
    ])

    # Convert to Python lists (cheap in Polars) for our logic
    first_list = df_pl["first"].to_list()
    last_list  = df_pl["last"].to_list()

    # Normalize names (fast pure-Python; still parallelizable if needed, but inexpensive)
    first_clean, last_clean = process_names_batch_smart(first_list, last_list)

    # Parallel email generation
    emails = generate_emails_smart_batch_parallel(first_clean, last_clean, domain, workers=workers, chunk_size=chunk_size)

    # Back to Polars and write
    df_pl = df_pl.with_columns(pl.Series("email", emails))
    df_pl.write_csv(output_file)

    elapsed_time = time.time() - start_time
    print(f"\n✅ Emails generated and saved to {output_file}")
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")

# =========================================================
# ======================== CLI ============================
# =========================================================
if __name__ == "__main__":
    # Tune workers/chunk_size for your machine & data size
    process_csv("names.csv", "generated_emails.csv", domain="example.com", workers=None, chunk_size=200_000)

