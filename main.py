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
    random_nums = rng.choice([True, False], n)
    # numbers are strings when used, else empty string
    numbers = np.where(random_nums,
                       rng.integers(1, 10000, n).astype(str),
                       np.empty(n, dtype=object))
    numbers[~random_nums] = ""

    # 6 casing flags per row (fn, ln, fi, li, rfn, rln)
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
    def lower_if(arr, flags):
        out = np.empty(n, dtype=object)
        # slicing the appropriate stride
        it = flags
        for i, s in enumerate(arr):
            out[i] = (s.lower() if it[i] else s)
        return out

    fn_arr  = lower_if(first_arr, casing[0::6])
    ln_arr  = lower_if(last_arr,  casing[1::6])
    fi_arr  = lower_if(fi_arr,    casing[2::6])
    li_arr  = lower_if(li_arr,    casing[3::6])
    rfn_arr = lower_if(rfn_arr,   casing[4::6])
    rln_arr = lower_if(rln_arr,   casing[5::6])

    # 75 pattern functions (same logic preserved)
    patterns = [
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}.{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}.{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}.{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}.{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}.{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}.{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}.{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}.{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}_{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}_{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}_{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}_{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}_{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}_{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}_{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}_{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rfn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rfn}{rln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rfn}.{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}{rln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}.{rln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rfn}{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rfn}.{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}.{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rln}{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rln}.{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rln}{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rln}.{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}{rfn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}.{rfn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}.{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rfn}_{ln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}_{rln}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rfn}_{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}_{li}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rln}_{fn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{rln}_{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}_{rfn}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}_{fi}{num}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}{num}{ln}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}{num}{ln}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}{num}{li}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}{num}{li}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}{num}{fn}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}{num}{fi}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}{num}{fn}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}{num}{fi}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}.{num}.{ln}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}.{num}.{ln}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}.{num}.{li}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}.{num}.{li}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}.{num}.{fn}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}.{num}.{fi}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}.{num}.{fn}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}.{num}.{fi}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}_{num}_{ln}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}_{num}_{ln}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fn}_{num}_{li}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{fi}_{num}_{li}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}_{num}_{fn}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{ln}_{num}_{fi}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}_{num}_{fn}",
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}_{num}_{fi}",
    ]

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
