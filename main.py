import random
import numpy as np
import re
import pandas as pd
import unicodedata  # <-- added for accent normalization
import os  # <-- needed to check & remove file

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
    Convert accented letters to plain ASCII equivalents.
    E.g., Á -> A, Ñ -> N
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", name) if c.isascii() and c.isalpha()
    )

def generate_email_patterns(first_name: str, last_name: str, domain: str) -> str:
    """
    Generate ONE plausible corporate email address based on the spec.
    """
    # ------------- 1) Validation & string casting -------------
    first_name = normalize_name(str(first_name))
    last_name  = normalize_name(str(last_name))

    # Allow empty strings (fallback handles), but if non-empty they must be ASCII A-Z only.
    if (first_name != "" and not _ASCII_ALPHA.fullmatch(first_name)):
        return np.nan
    if (last_name  != "" and not _ASCII_ALPHA.fullmatch(last_name)):
        return np.nan

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
# =============== Fast CSV Chunk Processing ===============
# =========================================================
def process_csv(input_file: str,
                output_file: str = "generated_emails.csv",
                domain: str = "example.com",
                chunksize: int = 100_000) -> None:
    """
    Streams a large CSV (expects columns: 'first','last') and writes 'email' column.
    If the output file already exists, it will be removed first.
    """
    # Remove existing file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    header_written = False

    for chunk in pd.read_csv(
        input_file,
        chunksize=chunksize,
        usecols=["first", "last"],
        dtype={"first": "string", "last": "string"},
        keep_default_na=True
    ):
        chunk["first"] = chunk["first"].fillna("")
        chunk["last"]  = chunk["last"].fillna("")

        emails = [generate_email_patterns(f, l, domain) for f, l in zip(chunk["first"].tolist(),
                                                                        chunk["last"].tolist())]
        chunk["email"] = emails

        chunk.to_csv(
            output_file,
            mode="a",  # always append since file was removed at start
            index=False,
            header=not header_written
        )
        header_written = True

    print(f"✅ Emails generated and saved to {output_file}")


# =========================================================
# ======================== CLI ============================
# =========================================================
if __name__ == "__main__":
    process_csv("names.csv", "generated_emails.csv", domain="example.com", chunksize=100_000)
