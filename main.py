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
    name = str(name)
    if not name:
        return ""
    return "".join(random.sample(name, min(len(name), count)))


# =========================================================
# =============== Required Main Function ==================
# =========================================================
_ASCII_ALPHA = re.compile(r"^[A-Za-z]+$")

def normalize_name(name: str) -> str:
    """Normalize accented names and remove non-letters."""
    if not name:
        return ""
    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if c.isalpha())


# =========================================================
# =============== SMART BATCH PROCESSING ==================
# =========================================================
def process_names_batch_smart(first_names, last_names):
    """Smart batch name processing without memory issues."""
    first_clean, last_clean = [], []

    for first, last in zip(first_names, last_names):
        first = str(first) if first else ""
        last  = str(last) if last else ""

        first = "".join(c for c in unicodedata.normalize("NFKD", first) if c.isalpha())
        last  = "".join(c for c in unicodedata.normalize("NFKD", last) if c.isalpha())

        if not first:
            first = "user"
        if not last:
            last = "user"

        first_clean.append(first)
        last_clean.append(last)

    return first_clean, last_clean


def generate_emails_smart_batch(first_names, last_names, domain):
    """Smart batch email generation with pre-generated randomness."""
    n = len(first_names)

    random_nums = np.random.choice([True, False], n)
    numbers = np.where(random_nums, np.random.randint(1, 10000, n), np.array([''] * n))

    casing = np.random.choice([True, False], n * 6)
    pattern_indices = np.random.randint(0, 75, n)   # 75 patterns
    letter_indices = np.random.randint(0, 10, n * 4)

    fallback_initials = ['a', 'b', 'c', 'd']
    fi_indices = np.random.randint(0, 4, n)
    li_indices = np.random.randint(0, 4, n)

    emails = []

    for i in range(n):
        first = first_names[i]
        last  = last_names[i]

        # Empty names fallback
        if not first and not last:
            fi = fallback_initials[fi_indices[i]]
            li = fallback_initials[li_indices[i]]
            num = str(numbers[i]) if numbers[i] != '' else ""
            emails.append(f"{fi}{li}{num}@{domain}")
            continue

        fi = first[0] if first else fallback_initials[fi_indices[i]]
        li = last[0] if last else fallback_initials[li_indices[i]]

        if len(first) >= 2:
            idx1 = letter_indices[i * 4] % len(first)
            idx2 = letter_indices[i * 4 + 1] % len(first)
            rfn = first[idx1] + first[idx2]
        else:
            rfn = first

        if len(last) >= 2:
            idx1 = letter_indices[i * 4 + 2] % len(last)
            idx2 = letter_indices[i * 4 + 3] % len(last)
            rln = last[idx1] + last[idx2]
        else:
            rln = last

        idx = i * 6
        fn  = first.lower() if casing[idx] else first
        ln  = last.lower()  if casing[idx + 1] else last
        fi  = fi.lower()    if casing[idx + 2] else fi
        li  = li.lower()    if casing[idx + 3] else li
        rfn = rfn.lower()   if casing[idx + 4] else rfn
        rln = rln.lower()   if casing[idx + 5] else rln

        num = str(numbers[i]) if numbers[i] != '' else ""

        # Select pattern (75 possibilities)
        p = pattern_indices[i]
        if p == 0:   local = f"{fn}{num}"
        elif p == 1: local = f"{ln}{num}"
        elif p == 2: local = f"{fn}{ln}{num}"
        elif p == 3: local = f"{fi}{ln}{num}"
        elif p == 4: local = f"{fn}{li}{num}"
        elif p == 5: local = f"{fi}{li}{num}"
        elif p == 6: local = f"{ln}{fn}{num}"
        elif p == 7: local = f"{ln}{fi}{num}"
        elif p == 8: local = f"{li}{fn}{num}"
        elif p == 9: local = f"{li}{fi}{num}"
        elif p == 10: local = f"{fn}.{ln}{num}"
        elif p == 11: local = f"{fi}.{ln}{num}"
        elif p == 12: local = f"{fn}.{li}{num}"
        elif p == 13: local = f"{fi}.{li}{num}"
        elif p == 14: local = f"{ln}.{fn}{num}"
        elif p == 15: local = f"{ln}.{fi}{num}"
        elif p == 16: local = f"{li}.{fn}{num}"
        elif p == 17: local = f"{li}.{fi}{num}"
        elif p == 18: local = f"{fn}_{ln}{num}"
        elif p == 19: local = f"{fi}_{ln}{num}"
        elif p == 20: local = f"{fn}_{li}{num}"
        elif p == 21: local = f"{fi}_{li}{num}"
        elif p == 22: local = f"{ln}_{fn}{num}"
        elif p == 23: local = f"{ln}_{fi}{num}"
        elif p == 24: local = f"{li}_{fn}{num}"
        elif p == 25: local = f"{li}_{fi}{num}"
        elif p == 26: local = f"{rfn}{num}"
        elif p == 27: local = f"{rln}{num}"
        elif p == 28: local = f"{rfn}{rln}{num}"
        elif p == 29: local = f"{rfn}.{ln}{num}"
        elif p == 30: local = f"{fi}{rln}{num}"
        elif p == 31: local = f"{fi}.{rln}{num}"
        elif p == 32: local = f"{rfn}{li}{num}"
        elif p == 33: local = f"{rfn}.{li}{num}"
        elif p == 34: local = f"{fi}{li}{num}"
        elif p == 35: local = f"{fi}.{li}{num}"
        elif p == 36: local = f"{rln}{fn}{num}"
        elif p == 37: local = f"{rln}.{fn}{num}"
        elif p == 38: local = f"{rln}{fi}{num}"
        elif p == 39: local = f"{rln}.{fi}{num}"
        elif p == 40: local = f"{li}{rfn}{num}"
        elif p == 41: local = f"{li}.{rfn}{num}"
        elif p == 42: local = f"{li}{fi}{num}"
        elif p == 43: local = f"{li}.{fi}{num}"
        elif p == 44: local = f"{rfn}_{ln}{num}"
        elif p == 45: local = f"{fi}_{rln}{num}"
        elif p == 46: local = f"{rfn}_{li}{num}"
        elif p == 47: local = f"{fi}_{li}{num}"
        elif p == 48: local = f"{rln}_{fn}{num}"
        elif p == 49: local = f"{rln}_{fi}{num}"
        elif p == 50: local = f"{li}_{rfn}{num}"
        elif p == 51: local = f"{li}_{fi}{num}"
        elif p == 52: local = f"{fn}{num}{ln}"
        elif p == 53: local = f"{fi}{num}{ln}"
        elif p == 54: local = f"{fn}{num}{li}"
        elif p == 55: local = f"{fi}{num}{li}"
        elif p == 56: local = f"{ln}{num}{fn}"
        elif p == 57: local = f"{ln}{num}{fi}"
        elif p == 58: local = f"{li}{num}{fn}"
        elif p == 59: local = f"{li}{num}{fi}"
        elif p == 60: local = f"{fn}.{num}.{ln}"
        elif p == 61: local = f"{fi}.{num}.{ln}"
        elif p == 62: local = f"{fn}.{num}.{li}"
        elif p == 63: local = f"{fi}.{num}.{li}"
        elif p == 64: local = f"{ln}.{num}.{fn}"
        elif p == 65: local = f"{ln}.{num}.{fi}"
        elif p == 66: local = f"{li}.{num}.{fn}"
        elif p == 67: local = f"{li}.{num}.{fi}"
        elif p == 68: local = f"{fn}_{num}_{ln}"
        elif p == 69: local = f"{fi}_{num}_{ln}"
        elif p == 70: local = f"{fn}_{num}_{li}"
        elif p == 71: local = f"{fi}_{num}_{li}"
        elif p == 72: local = f"{ln}_{num}_{fn}"
        elif p == 73: local = f"{ln}_{num}_{fi}"
        elif p == 74: local = f"{li}_{num}_{fn}"
        else:        local = f"{li}_{num}_{fi}"

        emails.append(f"{local}@{domain}")

    return emails


# =========================================================
# ================= MAIN CSV PROCESSOR ====================
# =========================================================
def process_csv(input_file: str,
                output_file: str = "generated_emails.csv",
                domain: str = "example.com") -> None:
    """High-performance CSV processing using Polars only."""
    start_time = time.time()

    if os.path.exists(output_file):
        os.remove(output_file)

    print("🚀 POLARS fast path: single read/process/write...")
    df_pl = pl.read_csv(input_file, columns=["first", "last"], infer_schema_length=0)
    df_pl = df_pl.with_columns([
        pl.col("first").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("last").cast(pl.Utf8, strict=False).fill_null("")
    ])

    first_list = df_pl["first"].to_list()
    last_list  = df_pl["last"].to_list()

    first_clean, last_clean = process_names_batch_smart(first_list, last_list)
    emails = generate_emails_smart_batch(first_clean, last_clean, domain)

    out_pl = df_pl.with_columns(pl.Series("email", emails))
    out_pl.write_csv(output_file)

    elapsed_time = time.time() - start_time
    print(f"\n✅ Emails generated and saved to {output_file}")
    print(f"⏱️ Completed in {elapsed_time:.2f} seconds")


# =========================================================
# ======================== CLI ============================
# =========================================================
if __name__ == "__main__":
    process_csv("names.csv", "generated_emails.csv", domain="example.com")
