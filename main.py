import random
import numpy as np
import re
import pandas as pd
import unicodedata  # <-- added for accent normalization
import os  # <-- needed to check & remove file
import time

# =========================================================
# =============== Required Helper Function ================
# =========================================================
def get_random_letters(name: str, count: int = 2) -> str:
    """
    Return a string formed by a random sample (no replacement)
    of up to `count` characters from name.
    Length is min(len(str(name)), count). Uses random.sample.
    """
    name = str(name)
    if not name:
        return ""
    return "".join(random.sample(name, min(len(name), count)))


# =========================================================
# =============== Required Main Function ==================
# =========================================================
# Keep pattern logic the same, but allow accent normalization
_ASCII_ALPHA = re.compile(r"^[A-Za-z]+$")

def normalize_name(name: str) -> str:
    """
    Convert accented letters to plain ASCII equivalents and clean up formatting.
    E.g., Á -> A, Ñ -> N, "A-DAN" -> "ADAN", "MELO CAM" -> "MELOCAM"
    """
    if not name:
        return ""
    
    # Convert to string and normalize unicode
    name = str(name)
    
    # Normalize unicode characters (accents, etc.)
    name = unicodedata.normalize("NFKD", name)
    
    # Remove all non-alphabetic characters (spaces, hyphens, dots, etc.)
    # Keep only A-Z and a-z characters
    cleaned = "".join(c for c in name if c.isalpha())
    
    return cleaned

def generate_email_patterns(first_name: str, last_name: str, domain: str) -> str:
    """
    Generate ONE plausible corporate email address based on the spec.
    """
    # ------------- 1) Validation & string casting -------------
    first_name = normalize_name(str(first_name))
    last_name  = normalize_name(str(last_name))

    # If names are empty after normalization, use fallback initials
    if not first_name and not last_name:
        # Both names are empty/invalid, generate random email
        fi = random.choice(["a", "b", "c", "d"])
        li = random.choice(["a", "b", "c", "d"])
        num = str(random.randint(1, 9999)) if random.getrandbits(1) else ""
        return f"{fi}{li}{num}@{domain}"
    
    # If only one name is empty, use the other one
    if not first_name:
        first_name = "user"
    if not last_name:
        last_name = "user"

    # ------------- 2) Initials (with fallback if empty) -------------
    fi = first_name[0] if first_name else random.choice(["a", "b", "c", "d"])
    li = last_name[0]  if last_name  else random.choice(["a", "b", "c", "d"])

    # ------------- 3) Random letters (no replacement) -------------
    rfn = get_random_letters(first_name)
    rln = get_random_letters(last_name)

    # ------------- 4) Random casing & optional number -------------
    def maybe_lower(s: str) -> str:
        return s.lower() if random.getrandbits(1) else s

    fn = maybe_lower(first_name)
    ln = maybe_lower(last_name)
    fi = maybe_lower(fi)
    li = maybe_lower(li)
    rfn = maybe_lower(rfn)
    rln = maybe_lower(rln)

    num = str(random.randint(1, 9999)) if random.getrandbits(1) else ""

    # ------------- 5) Pattern pool (EXACT families required) -------------
    patterns = {
        "fn":     f"{fn}{num}",
        "ln":     f"{ln}{num}",
        "fnln":   f"{fn}{ln}{num}",
        "filn":   f"{fi}{ln}{num}",
        "fnli":   f"{fn}{li}{num}",
        "fili":   f"{fi}{li}{num}",
        "lnfn":   f"{ln}{fn}{num}",
        "lnfi":   f"{ln}{fi}{num}",
        "lifn":   f"{li}{fn}{num}",
        "lifi":   f"{li}{fi}{num}",

        "fn.ln":  f"{fn}.{ln}{num}",
        "fi.ln":  f"{fi}.{ln}{num}",
        "fn.li":  f"{fn}.{li}{num}",
        "fi.li":  f"{fi}.{li}{num}",
        "ln.fn":  f"{ln}.{fn}{num}",
        "ln.fi":  f"{ln}.{fi}{num}",
        "li.fn":  f"{li}.{fn}{num}",
        "li.fi":  f"{li}.{fi}{num}",

        "fn_ln":  f"{fn}_{ln}{num}",
        "fi_ln":  f"{fi}_{ln}{num}",
        "fn_li":  f"{fn}_{li}{num}",
        "fi_li":  f"{fi}_{li}{num}",
        "ln_fn":  f"{ln}_{fn}{num}",
        "ln_fi":  f"{ln}_{fi}{num}",
        "li_fn":  f"{li}_{fn}{num}",
        "li_fi":  f"{li}_{fi}{num}",

        "rfn":     f"{rfn}{num}",
        "rln":     f"{rln}{num}",
        "rfnln":   f"{rfn}{rln}{num}",
        "rfn.ln":  f"{rfn}.{ln}{num}",
        "rfiln":   f"{fi}{rln}{num}",
        "rfi.ln":  f"{fi}.{rln}{num}",
        "rfnli":   f"{rfn}{li}{num}",
        "rfn.li":  f"{rfn}.{li}{num}",
        "rfili":   f"{fi}{li}{num}",
        "rfi.li":  f"{fi}.{li}{num}",
        "rlnfn":   f"{rln}{fn}{num}",
        "rln.fn":  f"{rln}.{fn}{num}",
        "rlnfi":   f"{rln}{fi}{num}",
        "rln.fi":  f"{rln}.{fi}{num}",
        "rlifn":   f"{li}{rfn}{num}",
        "rli.fn":  f"{li}.{rfn}{num}",
        "rlifi":   f"{li}{fi}{num}",
        "rli.fi":  f"{li}.{fi}{num}",

        "rfn_ln":  f"{rfn}_{ln}{num}",
        "rfi_ln":  f"{fi}_{rln}{num}",
        "rfn_li":  f"{rfn}_{li}{num}",
        "rfi_li":  f"{fi}_{li}{num}",
        "rln_fn":  f"{rln}_{fn}{num}",
        "rln_fi":  f"{rln}_{fi}{num}",
        "rli_fn":  f"{li}_{rfn}{num}",
        "rli_fi":  f"{li}_{fi}{num}",

        "fnnumln": f"{fn}{num}{ln}",
        "finumln": f"{fi}{num}{ln}",
        "fnnumli": f"{fn}{num}{li}",
        "finumli": f"{fi}{num}{li}",
        "lnnumfn": f"{ln}{num}{fn}",
        "lnnumfi": f"{ln}{num}{fi}",
        "linumfn": f"{li}{num}{fn}",
        "linumfi": f"{li}{num}{fi}",

        "fn.num.ln": f"{fn}.{num}.{ln}",
        "fi.num.ln": f"{fi}.{num}.{ln}",
        "fn.num.li": f"{fn}.{num}.{li}",
        "fi.num.li": f"{fi}.{num}.{li}",
        "ln.num.fn": f"{ln}.{num}.{fn}",
        "ln.num.fi": f"{ln}.{num}.{fi}",
        "li.num.fn": f"{li}.{num}.{fn}",
        "li.num.fi": f"{li}.{num}.{fi}",

        "fn_num_ln": f"{fn}_{num}_{ln}",
        "fi_num_ln": f"{fi}_{num}_{ln}",
        "fn_num_li": f"{fn}_{num}_{li}",
        "fi_num_li": f"{fi}_{num}_{li}",
        "ln_num_fn": f"{ln}_{num}_{fn}",
        "ln_num_fi": f"{ln}_{num}_{fi}",
        "li_num_fn": f"{li}_{num}_{fn}",
        "li_num_fi": f"{li}_{num}_{fi}",
    }

    local = random.choice(list(patterns.values()))
    return f"{local}@{domain}"


# =========================================================
# =============== SMART BATCH PROCESSING ==================
# =========================================================
def process_names_batch_smart(first_names, last_names):
    """Smart batch name processing without memory issues"""
    first_clean = []
    last_clean = []
    
    for first, last in zip(first_names, last_names):
        # Fast string handling
        if first is None:
            first = ""
        if last is None:
            last = ""
        
        first = str(first)
        last = str(last)
        
        # Fast normalization (inline for speed)
        if first:
            first_chars = []
            for c in unicodedata.normalize("NFKD", first):
                if c.isalpha():
                    first_chars.append(c)
            first = "".join(first_chars)
        
        if last:
            last_chars = []
            for c in unicodedata.normalize("NFKD", last):
                if c.isalpha():
                    last_chars.append(c)
            last = "".join(last_chars)
        
        # Handle empty names
        if not first:
            first = "user"
        if not last:
            last = "user"
        
        first_clean.append(first)
        last_clean.append(last)
    
    return first_clean, last_clean

def generate_emails_smart_batch(first_names, last_names, domain):
    """Smart batch email generation with pre-generated randomness"""
    n = len(first_names)
    
    # Pre-generate all random values for the entire batch
    random_nums = np.random.choice([True, False], n)
    numbers = np.where(random_nums, np.random.randint(1, 10000, n), np.array([''] * n))
    
    # Pre-generate random casing decisions (6 per name)
    casing = np.random.choice([True, False], n * 6)
    
    # Pre-generate random pattern indices
    pattern_indices = np.random.randint(0, 60, n)
    
    # Pre-generate random letter indices
    letter_indices = np.random.randint(0, 10, n * 4)  # 4 random letters per name
    
    # Pre-generate fallback initials
    fallback_initials = ['a', 'b', 'c', 'd']
    fi_indices = np.random.randint(0, 4, n)
    li_indices = np.random.randint(0, 4, n)
    
    emails = []
    
    for i in range(n):
        first = first_names[i]
        last = last_names[i]
        
        # Handle completely empty names
        if not first and not last:
            fi = fallback_initials[fi_indices[i]]
            li = fallback_initials[li_indices[i]]
            num = str(numbers[i]) if numbers[i] != '' else ""
            emails.append(f"{fi}{li}{num}@{domain}")
            continue
        
        # Get initials
        fi = first[0] if first else fallback_initials[fi_indices[i]]
        li = last[0] if last else fallback_initials[li_indices[i]]
        
        # Random letters (using pre-generated indices)
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
        
        # Random casing using pre-generated decisions
        idx = i * 6
        fn = first.lower() if casing[idx] else first
        ln = last.lower() if casing[idx + 1] else last
        fi = fi.lower() if casing[idx + 2] else fi
        li = li.lower() if casing[idx + 3] else li
        rfn = rfn.lower() if casing[idx + 4] else rfn
        rln = rln.lower() if casing[idx + 5] else rln
        
        # Get number
        num = str(numbers[i]) if numbers[i] != '' else ""
        
        # Pre-built patterns array
        patterns = [
            f"{fn}{num}", f"{ln}{num}", f"{fn}{ln}{num}", f"{fi}{ln}{num}",
            f"{fn}{li}{num}", f"{fi}{li}{num}", f"{ln}{fn}{num}", f"{ln}{fi}{num}",
            f"{li}{fn}{num}", f"{li}{fi}{num}", f"{fn}.{ln}{num}", f"{fi}.{ln}{num}",
            f"{fn}.{li}{num}", f"{fi}.{li}{num}", f"{ln}.{fn}{num}", f"{ln}.{fi}{num}",
            f"{li}.{fn}{num}", f"{li}.{fi}{num}", f"{fn}_{ln}{num}", f"{fi}_{ln}{num}",
            f"{fn}_{li}{num}", f"{fi}_{li}{num}", f"{ln}_{fn}{num}", f"{ln}_{fi}{num}",
            f"{li}_{fn}{num}", f"{li}_{fi}{num}", f"{rfn}{num}", f"{rln}{num}",
            f"{rfn}{rln}{num}", f"{rfn}.{ln}{num}", f"{fi}{rln}{num}", f"{fi}.{rln}{num}",
            f"{rfn}{li}{num}", f"{rfn}.{li}{num}", f"{fi}{li}{num}", f"{fi}.{li}{num}",
            f"{rln}{fn}{num}", f"{rln}.{fn}{num}", f"{rln}{fi}{num}", f"{rln}.{fi}{num}",
            f"{li}{rfn}{num}", f"{li}.{rfn}{num}", f"{li}{fi}{num}", f"{li}.{fi}{num}",
            f"{rfn}_{ln}{num}", f"{fi}_{rln}{num}", f"{rfn}_{li}{num}", f"{fi}_{li}{num}",
            f"{rln}_{fn}{num}", f"{rln}_{fi}{num}", f"{li}_{rfn}{num}", f"{li}_{fi}{num}",
            f"{fn}{num}{ln}", f"{fi}{num}{ln}", f"{fn}{num}{li}", f"{fi}{num}{li}",
            f"{ln}{num}{fn}", f"{ln}{num}{fi}", f"{li}{num}{fn}", f"{li}{num}{fi}",
            f"{fn}.{num}.{ln}", f"{fi}.{num}.{ln}", f"{fn}.{num}.{li}", f"{fi}.{num}.{li}",
            f"{ln}.{num}.{fn}", f"{ln}.{num}.{fi}", f"{li}.{num}.{fn}", f"{li}.{num}.{fi}",
            f"{fn}_{num}_{ln}", f"{fi}_{num}_{ln}", f"{fn}_{num}_{li}", f"{fi}_{num}_{li}",
            f"{ln}_{num}_{fn}", f"{ln}_{num}_{fi}", f"{li}_{num}_{fn}", f"{li}_{num}_{fi}"
        ]
        
        # Use pre-generated pattern index
        emails.append(f"{patterns[pattern_indices[i]]}@{domain}")
    
    return emails

def process_csv(input_file: str,
                output_file: str = "generated_emails.csv",
                domain: str = "example.com",
                chunksize: int = 200_000) -> None:
    """
    Smart batch CSV processing that actually works without memory issues.
    Target: Under 1 minute for 300MB file.
    """
    start_time = time.time()
    last_report = start_time

    # Remove existing file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    header_written = False
    
    print(f"🚀 SMART BATCH mode: {chunksize:,} row chunks with optimized processing...")
    
    # Process with optimized chunks
    for chunk_num, chunk in enumerate(pd.read_csv(
        input_file,
        chunksize=chunksize,
        usecols=["first", "last"],
        dtype={"first": "string", "last": "string"},
        keep_default_na=True,
        engine='c'
    ), start=1):
        
        # Fill NaN values efficiently
        chunk["first"] = chunk["first"].fillna("")
        chunk["last"] = chunk["last"].fillna("")
        
        # Smart batch name processing
        first_clean, last_clean = process_names_batch_smart(chunk["first"].tolist(), chunk["last"].tolist())
        
        # Smart batch email generation
        emails = generate_emails_smart_batch(first_clean, last_clean, domain)
        chunk["email"] = emails
        
        # Write chunk efficiently
        chunk.to_csv(
            output_file,
            mode="a",
            index=False,
            header=not header_written,
            float_format='%.0f'
        )
        header_written = True
        
        # Progress update
        now = time.time()
        if now - last_report >= 3:
            elapsed = now - start_time
            print(f"⏳ Processing... {elapsed:.2f} seconds elapsed (after {chunk_num} chunks)")
            last_report = now

    elapsed_time = time.time() - start_time
    print(f"\n✅ Emails generated and saved to {output_file}")
    print(f"⏱️ Completed in {elapsed_time:.2f} seconds")


# =========================================================
# ======================== CLI ============================
# =========================================================
if __name__ == "__main__":
    process_csv("names.csv", "generated_emails.csv", domain="example.com", chunksize=200_000)