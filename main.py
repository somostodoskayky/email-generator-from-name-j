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

    # Always lowercase all name components for consistent email format
    fn_arr  = np.array([s.lower() for s in first_arr], dtype=object)
    ln_arr  = np.array([s.lower() for s in last_arr], dtype=object)
    fi_arr  = np.array([s.lower() for s in fi_arr], dtype=object)
    li_arr  = np.array([s.lower() for s in li_arr], dtype=object)
    rfn_arr = np.array([s.lower() for s in rfn_arr], dtype=object)
    rln_arr = np.array([s.lower() for s in rln_arr], dtype=object)

    # Compact long names (>4 chars) to a 3-letter random sample (no replacement)
    def compact_array(arr):
        out = np.empty(n, dtype=object)
        for i, s in enumerate(arr):
            if len(s) > 4:
                k = 3 if len(s) >= 3 else len(s)
                idxs = rng.choice(len(s), size=k, replace=False)
                out[i] = "".join(s[j] for j in idxs)
            else:
                out[i] = s
        return out

    fnc_arr = compact_array(fn_arr)
    lnc_arr = compact_array(ln_arr)

    # 75 pattern functions (same logic preserved)
    def safe_concat(*parts):
        """Safely concatenate parts, avoiding double special characters."""
        result = ""
        for part in parts:
            if part:  # Only add non-empty parts
                if result and result[-1] in '._' and part[0] in '._':
                    # Avoid double special characters
                    result += part[1:] if len(part) > 1 else ""
                else:
                    result += part
        return result

    patterns = [
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ".", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ".", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, ".", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, ".", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, "_", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, "_", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, "_", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, rln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, ".", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, rln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", rln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, ".", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rln, fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rln, ".", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rln, fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rln, ".", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, rfn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", rfn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, "_", ln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", rln, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rfn, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", li, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rln, "_", fn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(rln, "_", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", rfn, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", fi, num),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, num, ln),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, num, ln),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, num, li),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, num, li),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, num, fn),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, num, fi),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, num, fn),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, num, fi),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ".", num, ".", ln),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", num, ".", ln),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, ".", num, ".", li),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, ".", num, ".", li),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, ".", num, ".", fn),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, ".", num, ".", fi),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", num, ".", fn),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, ".", num, ".", fi),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, "_", num, "_", ln),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", num, "_", ln),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fn, "_", num, "_", li),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(fi, "_", num, "_", li),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, "_", num, "_", fn),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(ln, "_", num, "_", fi),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", num, "_", fn),
        lambda fn, ln, fi, li, rfn, rln, num: safe_concat(li, "_", num, "_", fi),
    ]

    # Build locals per pattern (single Python loop over rows)
    local_parts = np.empty(n, dtype=object)
    for i in range(n):
        if (first_arr[i] or last_arr[i]):
            p = pattern_indices[i]
            local_parts[i] = patterns[p](fnc_arr[i], lnc_arr[i], fi_arr[i], li_arr[i], rfn_arr[i], rln_arr[i], numbers[i])
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
    print(f"⏱️ Completed in {time.time() - start_time:.2f} seconds")

    # Parallel email generation
    emails = generate_emails_smart_batch_parallel(first_clean, last_clean, domain, workers=workers, chunk_size=chunk_size)
    print(f"⏱️ Completed in {time.time() - start_time:.2f} seconds")

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
