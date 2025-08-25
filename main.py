import random
import numpy as np
import re
import polars as pl
import unicodedata
import os
import time

# =========================================================
# =============== Required Helper Function ================
# =========================================================
def get_random_letters(name: str, count: int = 2) -> str:
    """Return up to `count` random letters from name (no replacement)."""
    if not name:
        return ""
    name = str(name)
    n = len(name)
    if n <= count:
        return name
    # Use random.sample efficiently
    return "".join(random.sample(name, count))


# =========================================================
# =============== Required Main Function ==================
# =========================================================
def normalize_name(name: str) -> str:
    """Normalize accented names and remove non-letters."""
    if not name:
        return ""
    name = str(name)
    # NFKD + filter alpha, faster with join and generator
    return "".join(c for c in unicodedata.normalize("NFKD", name) if c.isalpha())


# =========================================================
# =============== SMART BATCH PROCESSING ==================
# =========================================================
def process_names_batch_smart(first_names, last_names):
    """Smart batch name processing without memory issues (Polars compatible)."""
    # Use list comprehension + normalize_name function (much faster)
    first_clean = [normalize_name(x) or "user" for x in first_names]
    last_clean  = [normalize_name(x) or "user" for x in last_names]
    return first_clean, last_clean


def generate_emails_smart_batch(first_names, last_names, domain):
    """Optimized memory-safe smart batch email generation with pattern lookup."""

    n = len(first_names)

    first_arr = np.array(first_names, dtype=object)
    last_arr  = np.array(last_names, dtype=object)
    fallback_initials = np.array(['a', 'b', 'c', 'd'])

    # ---------------- Precompute randomness ----------------
    random_nums = np.random.choice([True, False], n)
    numbers = np.where(random_nums, np.random.randint(1, 10000, n).astype(str), np.array(['']*n))

    casing = np.random.choice([True, False], n*6)
    pattern_indices = np.random.randint(0, 75, n)
    letter_indices = np.random.randint(0, 10, n*4)

    fi_indices = np.random.randint(0, 4, n)
    li_indices = np.random.randint(0, 4, n)

    # ---------------- Compute first & last initials ----------------
    fi_arr = np.array([f[0] if f else fallback_initials[fi_indices[i]] for i, f in enumerate(first_arr)], dtype=object)
    li_arr = np.array([l[0] if l else fallback_initials[li_indices[i]] for i, l in enumerate(last_arr)], dtype=object)

    # ---------------- Compute random 2-letter combos ----------------
    def random_pair(arr, idx_base):
        res = []
        for i, s in enumerate(arr):
            if len(s) >= 2:
                idx1 = letter_indices[idx_base[i]] % len(s)
                idx2 = letter_indices[idx_base[i]+1] % len(s)
                res.append(s[idx1] + s[idx2])
            else:
                res.append(s)
        return np.array(res, dtype=object)

    li_idx = np.arange(n) * 4
    rfn_arr = random_pair(first_arr, li_idx)
    rln_arr = random_pair(last_arr, li_idx + 2)

    # ---------------- Apply casing safely ----------------
    fn_arr  = np.array([s.lower() if c else s for s, c in zip(first_arr, casing[0::6])], dtype=object)
    ln_arr  = np.array([s.lower() if c else s for s, c in zip(last_arr, casing[1::6])], dtype=object)
    fi_arr  = np.array([s.lower() if c else s for s, c in zip(fi_arr, casing[2::6])], dtype=object)
    li_arr  = np.array([s.lower() if c else s for s, c in zip(li_arr, casing[3::6])], dtype=object)
    rfn_arr = np.array([s.lower() if c else s for s, c in zip(rfn_arr, casing[4::6])], dtype=object)
    rln_arr = np.array([s.lower() if c else s for s, c in zip(rln_arr, casing[5::6])], dtype=object)

    # ---------------- Precompute 75 pattern functions ----------------
    # Each lambda takes fn, ln, fi, li, rfn, rln, num and returns local part
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
        lambda fn, ln, fi, li, rfn, rln, num: f"{li}_{num}_{fi}"
    ]

    # ---------------- Generate emails ----------------
    emails = np.array([
        patterns[p](fn_arr[i], ln_arr[i], fi_arr[i], li_arr[i], rfn_arr[i], rln_arr[i], numbers[i]) + f"@{domain}"
        if first_arr[i] or last_arr[i] else
        f"{fi_arr[i]}{li_arr[i]}{numbers[i]}@{domain}"
        for i, p in enumerate(pattern_indices)
    ], dtype=object)

    return emails


# =========================================================
# ================= MAIN CSV PROCESSOR ====================
# =========================================================
def process_csv(input_file: str,
                output_file: str = "generated_emails.csv",
                domain: str = "example.com") -> None:
    """High-performance CSV processing using Polars (logic preserved)."""
    start_time = time.time()

    if os.path.exists(output_file):
        os.remove(output_file)

    print("🚀 POLARS fast path (original email logic preserved)")

    # Read CSV (only needed columns)
    df_pl = pl.read_csv(input_file, columns=["first", "last"], infer_schema_length=0)

    # Clean columns
    df_pl = df_pl.with_columns([
        pl.col("first").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("last").cast(pl.Utf8, strict=False).fill_null("")
    ])

    # Convert once to Python lists for your custom logic
    first_list = df_pl["first"].to_list()
    last_list  = df_pl["last"].to_list()

    # Apply your original logic
    first_clean, last_clean = process_names_batch_smart(first_list, last_list)
    emails = generate_emails_smart_batch(first_clean, last_clean, domain)

    # Add emails back to Polars DataFrame
    df_pl = df_pl.with_columns(pl.Series("email", emails))

    # Write to CSV
    df_pl.write_csv(output_file)

    elapsed_time = time.time() - start_time
    print(f"\n✅ Emails generated and saved to {output_file}")
    print(f"⏱️ Completed in {elapsed_time:.2f} seconds")


# =========================================================
# ======================== CLI ============================
# =========================================================
if __name__ == "__main__":
    process_csv("names.csv", "generated_emails.csv", domain="example.com")
