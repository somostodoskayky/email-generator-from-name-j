import pandas as pd
import random
import unicodedata
import re
import numpy as np

# -------------------------------
# Normalize names (strip accents, keep A–Z)
# -------------------------------
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^A-Za-z]", "", name)


# -------------------------------
# Helper: pick up to count random letters
# -------------------------------
def get_random_letters(name: str, count: int = 2) -> str:
    if not name:
        return ""
    return "".join(random.sample(name, min(len(name), count)))


# -------------------------------
# Main email generator
# -------------------------------
def generate_email_patterns(first_name: str, last_name: str, domain: str) -> str:
    # Validation
    if not first_name.isalpha() or not last_name.isalpha():
        return np.nan

    # Initials
    fi = first_name[0] if first_name else random.choice(["a", "b", "c", "d"])
    li = last_name[0] if last_name else random.choice(["a", "b", "c", "d"])

    # Random-letters
    rfn = get_random_letters(first_name)
    rln = get_random_letters(last_name)

    # Random casing
    def maybe_lower(s): return s.lower() if random.choice([True, False]) else s
    fn, ln, fi, li, rfn, rln = map(maybe_lower, [first_name, last_name, fi, li, rfn, rln])

    # Optional number
    num = str(random.randint(1, 9999)) if random.choice([True, False]) else ""

    # --- Pattern pool (templates with placeholders) ---
    patterns = [
        "{fn}{num}", "{ln}{num}", "{fn}{ln}{num}",
        "{fn}.{ln}{num}", "{fi}{ln}{num}", "{fi}.{ln}{num}",
        "{fn}{li}{num}", "{fn}.{li}{num}", "{fi}{li}{num}", "{fi}.{li}{num}",
        "{ln}{fn}{num}", "{ln}.{fn}{num}", "{ln}{fi}{num}", "{ln}.{fi}{num}",
        "{li}{fn}{num}", "{li}.{fn}{num}", "{li}{fi}{num}", "{li}.{fi}{num}",

        "{fn}_{ln}{num}", "{fi}_{ln}{num}", "{fn}_{li}{num}", "{fi}_{li}{num}",
        "{ln}_{fn}{num}", "{ln}_{fi}{num}", "{li}_{fn}{num}", "{li}_{fi}{num}",

        "{rfn}{num}", "{rln}{num}", "{rfn}{rln}{num}", "{rfn}.{ln}{num}",
        "{fi}{rln}{num}", "{fi}.{rln}{num}", "{rfn}{li}{num}", "{rfn}.{li}{num}",
        "{fi}{li}{num}", "{fi}.{li}{num}", "{rln}{fn}{num}", "{rln}.{fn}{num}",
        "{rln}{fi}{num}", "{rln}.{fi}{num}", "{li}{rfn}{num}", "{li}.{rfn}{num}",
        "{li}{fi}{num}", "{li}.{fi}{num}",

        "{rfn}_{ln}{num}", "{fi}_{rln}{num}", "{rfn}_{li}{num}", "{fi}_{li}{num}",
        "{rln}_{fn}{num}", "{rln}_{fi}{num}", "{li}_{rfn}{num}", "{li}_{fi}{num}",

        "{fn}{num}{ln}", "{fi}{num}{ln}", "{fn}{num}{li}", "{fi}{num}{li}",
        "{ln}{num}{fn}", "{ln}{num}{fi}", "{li}{num}{fn}", "{li}{num}{fi}",

        "{fn}.{num}.{ln}", "{fi}.{num}.{ln}", "{fn}.{num}.{li}", "{fi}.{num}.{li}",
        "{ln}.{num}.{fn}", "{ln}.{num}.{fi}", "{li}.{num}.{fn}", "{li}.{num}.{fi}",

        "{fn}_{num}_{ln}", "{fi}_{num}_{ln}", "{fn}_{num}_{li}", "{fi}_{num}_{li}",
        "{ln}_{num}_{fn}", "{ln}_{num}_{fi}", "{li}_{num}_{fn}", "{li}_{num}_{fi}",
    ]

    # Pick one random template and format
    template = random.choice(patterns)
    local = template.format(fn=fn, ln=ln, fi=fi, li=li, rfn=rfn, rln=rln, num=num)

    return f"{local}@{domain}"


# -------------------------------
# Run for CSV file
# -------------------------------
if __name__ == "__main__":
    df = pd.read_csv("name.csv")

    emails = []
    for _, row in df.iterrows():
        fn = normalize_name(row["first"])
        ln = normalize_name(row["last"])
        emails.append(generate_email_patterns(fn, ln, "example.com"))

    df["email"] = emails
    df.to_csv("generated_emails.csv", index=False)
    print("✅ Emails generated and saved to generated_emails.csv")
