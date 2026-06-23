"""Build minimal-pair CSV datasets for all 6 phonetic operations.

Handles tokenization auditing (single-BPE-token targets, matching token counts),
generates CSVs in BLiMP format for the EAP-IG pipeline.

Usage:
    uv run python -m lib.build_datasets
    uv run python -m lib.build_datasets --ops op3_initialism op4_oronym
    uv run python -m lib.build_datasets --audit-only
"""
import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tok_id(word: str) -> int | None:
    """Return token ID if word is a single BPE token, else None."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def tok_len(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def audit_pair(clean: str, corrupted: str, target: str, foil: str) -> dict | None:
    """Returns a valid row dict or None if the pair fails tokenization audit."""
    target_id = tok_id(target)
    foil_id = tok_id(foil)
    if target_id is None:
        return None
    if foil_id is None:
        return None
    if tok_len(clean) != tok_len(corrupted):
        return None
    return {
        "clean": clean,
        "corrupted": corrupted,
        "target_idx": target_id,
        "foil_idx": foil_id,
    }


def write_csv(rows: list[dict], path: Path, task: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clean", "corrupted", "target_idx", "foil_idx"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {task}: {len(rows)} pairs -> {path}")


# ============================================================
# Op 1: Rhyming hypocorism (name -> nickname)
# ============================================================
OP1_PAIRS = [
    # (name, nickname, foil_nickname, template)
    # 1a: phonetically transparent
    ("Robert", "Bob", "Tom"),
    ("William", "Bill", "Phil"),
    ("William", "Will", "Phil"),
    ("Richard", "Dick", "Nick"),
    ("Richard", "Rick", "Nick"),
    ("Edward", "Ed", "Al"),
    ("Edward", "Ted", "Red"),
    ("Margaret", "Peggy", "Betty"),
    ("Elizabeth", "Liz", "Jill"),
    ("Elizabeth", "Beth", "Ruth"),
    ("Katherine", "Kate", "Jane"),
    ("Katherine", "Kathy", "Betty"),
    ("Patricia", "Pat", "Sal"),
    ("Jennifer", "Jen", "Kim"),
    ("Benjamin", "Ben", "Dan"),
    ("Nicholas", "Nick", "Rick"),
    ("Alexander", "Alex", "Mark"),
    ("Anthony", "Tony", "Joey"),
    ("Joseph", "Joe", "Dan"),
    ("Michael", "Mike", "Luke"),
    ("Thomas", "Tom", "Dan"),
    ("Daniel", "Dan", "Sam"),
    ("Samuel", "Sam", "Dan"),
    ("Jonathan", "Jon", "Tom"),
    ("Theodore", "Ted", "Ned"),
    ("Frederick", "Fred", "Brad"),
    ("Gregory", "Greg", "Brad"),
    ("Christopher", "Chris", "James"),
    ("Timothy", "Tim", "Jim"),
    ("Matthew", "Matt", "Jack"),
    ("Stephen", "Steve", "James"),
    ("Patrick", "Pat", "Matt"),
    ("Andrew", "Andy", "Jack"),
]

OP1_TEMPLATES = [
    ("People call {name} by the name", "People call {corrupt_name} by the name"),
]


def build_op1():
    rows = []
    names = [p[0] for p in OP1_PAIRS]
    for name, nickname, foil in OP1_PAIRS:
        for clean_tpl, corrupt_tpl in OP1_TEMPLATES:
            corrupt_names = [n for n in names if n != name and tok_len(n) == tok_len(name)]
            if not corrupt_names:
                corrupt_names = [n for n in names if n != name]
            for corrupt_name in corrupt_names[:4]:
                clean = clean_tpl.format(name=name)
                corrupted = corrupt_tpl.format(corrupt_name=corrupt_name)
                row = audit_pair(clean, corrupted, nickname, foil)
                if row:
                    rows.append(row)
    return rows


# ============================================================
# Op 2: Clipping (name -> first syllable)
# ============================================================
OP2_PAIRS = [
    # (full_name, clipped, foil)
    # Foil = phonetically distant short name, not a plausible clip of the full name.
    ("Timothy", "Tim", "Joe"),
    ("Benjamin", "Ben", "Joe"),
    ("Elizabeth", "Liz", "Joe"),
    ("Jennifer", "Jen", "Joe"),
    ("Nicholas", "Nick", "Joe"),
    ("Alexander", "Alex", "Joe"),
    ("Anthony", "Tony", "Joe"),
    ("Frederick", "Fred", "Joe"),
    ("Gregory", "Greg", "Joe"),
    ("Jonathan", "Jon", "Joe"),
    ("Michael", "Mike", "Joe"),
    ("Patrick", "Pat", "Joe"),
    ("Samantha", "Sam", "Joe"),
    ("Nathaniel", "Nate", "Joe"),
    ("Maximilian", "Max", "Joe"),
    ("Theodore", "Ted", "Joe"),
    ("Daniel", "Dan", "Joe"),
    ("Matthew", "Matt", "Joe"),
]

OP2_TEMPLATES = [
    ("Most people call {name}", "Most people call {corrupt_name}"),
]


def build_op2():
    rows = []
    names = [p[0] for p in OP2_PAIRS]
    for name, clip, foil in OP2_PAIRS:
        for clean_tpl, corrupt_tpl in OP2_TEMPLATES:
            corrupt_names = [n for n in names if n != name and tok_len(n) == tok_len(name)]
            if not corrupt_names:
                corrupt_names = [n for n in names if n != name]
            for corrupt_name in corrupt_names[:4]:
                clean = clean_tpl.format(name=name)
                corrupted = corrupt_tpl.format(corrupt_name=corrupt_name)
                row = audit_pair(clean, corrupted, clip, foil)
                if row:
                    rows.append(row)
    return rows


# ============================================================
# Op 3: Initialism (letter -> letter name)
# ============================================================
OP3_MAP = {
    # (target, foil) — foil is a phonetically DISTANT letter name
    "A": ("Ay", "Ex"),
    "B": ("Bee", "Why"),
    "C": ("See", "Why"),
    "D": ("Dee", "Why"),
    "F": ("Eff", "Why"),
    "I": ("Eye", "Why"),
    "J": ("Jay", "Why"),
    "K": ("Kay", "Why"),
    "L": ("El", "Why"),
    "M": ("Em", "Why"),
    "O": ("Oh", "Why"),
    "R": ("Are", "Why"),
    "S": ("Es", "Why"),
    "T": ("Tea", "Why"),
    "U": ("You", "Ex"),
    "X": ("Ex", "Oh"),
    "Y": ("Why", "Ex"),
}

OP3_TEMPLATES = [
    ("The letter {letter} is pronounced", "The letter {corrupt_letter} is pronounced"),
]


def build_op3():
    rows = []
    letters = list(OP3_MAP.keys())
    for letter, (name, foil) in OP3_MAP.items():
        for clean_tpl, corrupt_tpl in OP3_TEMPLATES:
            corrupt_letters = [l for l in letters if l != letter]
            for corrupt_letter in corrupt_letters[:4]:
                clean = clean_tpl.format(letter=letter)
                corrupted = corrupt_tpl.format(corrupt_letter=corrupt_letter)
                row = audit_pair(clean, corrupted, name, foil)
                if row:
                    rows.append(row)
    return rows


# ============================================================
# Op 4: Oronym (cross-boundary phoneme fusion)
# ============================================================
OP4_PAIRS = [
    # (first_name, last_name, target_word, foil_word, corrupt_first)
    # All targets+foils verified single-token in GPT-2, corrupt matched token length.
    # Foil = semantically related but phonetically unrelated word.
    # Original pun names
    ("Bill", "Ding", "building", "house", "Phil"),
    ("Curt", "Ain", "curtain", "window", "Bert"),
    ("Sandy", "Beach", "sandy", "dusty", "Randy"),
    ("Tim", "Burr", "timber", "lumber", "Jim"),
    ("Ginger", "Ale", "ginger", "pepper", "Singer"),
    ("Justin", "Time", "just", "only", "Austin"),
    ("Candy", "Barr", "candy", "sugar", "Sandy"),
    ("Richard", "Tater", "dictator", "tyrant", "Michael"),
    ("William", "Ding", "building", "house", "Jonathan"),
    # Constructed from verified single-token targets
    ("Gar", "Den", "garden", "yard", "Car"),
    ("Par", "Don", "pardon", "excuse", "Car"),
    ("Cap", "Tain", "captain", "colonel", "Map"),
    ("Fawn", "Tain", "fountain", "river", "John"),
    ("Mon", "Stir", "monster", "beast", "Don"),
    ("Pump", "Kin", "pumpkin", "squash", "Jump"),
    ("Doll", "Fin", "dolphin", "whale", "Toll"),
    ("Bay", "Con", "bacon", "ham", "Ray"),
    ("Lem", "On", "lemon", "orange", "Gem"),
    ("Drag", "On", "dragon", "beast", "Flag"),
    ("Or", "Gan", "organ", "piano", "For"),
    ("May", "Den", "maiden", "lady", "Ray"),
    ("Car", "Pet", "carpet", "floor", "Bar"),
    ("But", "Ter", "butter", "cream", "Cut"),
    ("Ham", "Mer", "hammer", "wrench", "Jam"),
    ("Must", "Ard", "mustard", "sauce", "Just"),
    ("Win", "Dow", "window", "door", "Fin"),
    ("Art", "Ist", "artist", "painter", "Bart"),
]

OP4_TEMPLATES = [
    ("The name {first} {last} sounds like the word", "The name {corrupt_first} {last} sounds like the word"),
]


CORRUPT_NAMES_POOL = [
    "Phil", "Bert", "Jim", "Sam", "Val", "Cal", "Mel", "Ned",
    "Rod", "Rex", "Roy", "Guy", "Max", "Bud", "Hal", "Earl",
    "Dean", "Grant", "Brad", "Todd", "Mark", "Luke", "Seth", "Joel",
]


def build_op4():
    rows = []
    firsts = [p[0] for p in OP4_PAIRS]
    all_corrupts = firsts + CORRUPT_NAMES_POOL
    for first, last, target, foil, _ in OP4_PAIRS:
        for clean_tpl, corrupt_tpl in OP4_TEMPLATES:
            clean = clean_tpl.format(first=first, last=last)
            cl = tok_len(clean)
            used = set()
            for corrupt_first in all_corrupts:
                if corrupt_first == first:
                    continue
                corrupted = corrupt_tpl.format(corrupt_first=corrupt_first, last=last)
                if tok_len(corrupted) != cl:
                    continue
                if corrupt_first in used:
                    continue
                row = audit_pair(clean, corrupted, target, foil)
                if row:
                    rows.append(row)
                    used.add(corrupt_first)
                if len(used) >= 4:
                    break
    return rows


# ============================================================
# Op 5: Homophone (word that sounds like a name)
# ============================================================
OP5_PAIRS = [
    # (name_word, homophone, foil)
    ("Neil", "kneel", "wheel"),
    ("Gene", "jean", "shirt"),
    ("Mark", "mark", "spot"),
    ("Grace", "grace", "charm"),
    ("Rose", "rose", "lily"),
    ("Bill", "bill", "coin"),
    ("Jack", "jack", "plug"),
    ("Frank", "frank", "honest"),
    ("Art", "art", "math"),
    ("Clay", "clay", "sand"),
    ("Dawn", "dawn", "dusk"),
    ("Dale", "dale", "hill"),
    ("Glen", "glen", "hill"),
    ("Hazel", "hazel", "brown"),
    ("Heath", "heath", "moor"),
    ("Holly", "holly", "ivy"),
    ("Iris", "iris", "lily"),
    ("Ivy", "ivy", "vine"),
    ("Jade", "jade", "ruby"),
    ("Jasper", "jasper", "stone"),
    ("Lily", "lily", "rose"),
    ("Olive", "olive", "grape"),
    ("Pearl", "pearl", "ruby"),
    ("Ruby", "ruby", "jade"),
    ("Violet", "violet", "purple"),
    ("Cliff", "cliff", "hill"),
    ("Robin", "robin", "eagle"),
    ("Lance", "lance", "sword"),
    ("Grant", "grant", "gift"),
    ("Wade", "wade", "swim"),
]

OP5_TEMPLATES = [
    ("The name {name} is also a word meaning", "The name {corrupt_name} is also a word meaning"),
]


def build_op5():
    rows = []
    names = [p[0] for p in OP5_PAIRS]
    for name, target, foil in OP5_PAIRS:
        for clean_tpl, corrupt_tpl in OP5_TEMPLATES:
            corrupt_names = [n for n in names if n != name and tok_len(n) == tok_len(name)]
            if not corrupt_names:
                corrupt_names = [n for n in names if n != name]
            for corrupt_name in corrupt_names[:4]:
                clean = clean_tpl.format(name=name)
                corrupted = corrupt_tpl.format(corrupt_name=corrupt_name)
                row = audit_pair(clean, corrupted, target, foil)
                if row:
                    rows.append(row)
    return rows


# ============================================================
# Op 6: Folk etymology (reverse decomposition)
# ============================================================
OP7_PAIRS = [
    # (word, target_name, foil_name, corrupt_word)
    # Op 6 = reverse decomposition: word -> first name hidden inside it.
    # Target MUST be a real name that GPT-2 would plausibly predict,
    # AND the name must genuinely appear in the word's spelling.
    ("timber", "Tim", "Joe", "lumber"),
    ("ginger", "Ginger", "Sarah", "pepper"),
    ("maiden", "May", "Joe", "knight"),
    ("artist", "Art", "Joe", "painter"),
    ("dolphin", "Don", "Joe", "whale"),
    ("window", "Win", "Joe", "mirror"),
    ("jacket", "Jack", "Joe", "coat"),
    ("golden", "Gold", "Jane", "silver"),
    ("marshal", "Mark", "Joe", "general"),
    ("dandy", "Dan", "Joe", "fancy"),
    ("hamlet", "Ham", "Joe", "village"),
    ("sandwich", "Sand", "Jane", "burger"),
    ("mandate", "Man", "Joe", "order"),
    ("marginal", "Marge", "Sarah", "central"),
    ("margin", "Mar", "Joe", "center"),
    ("gallant", "Gal", "Joe", "coward"),
    ("martini", "Martin", "Sarah", "whiskey"),
]

OP7_TEMPLATES = [
    ("The word {word} contains the name", "The word {corrupt_word} contains the name"),
]


CORRUPT_WORDS_POOL = [
    "table", "chair", "stone", "river", "glass", "metal", "brick",
    "cloud", "flame", "storm", "frost", "steel", "coral", "cedar",
    "maple", "ivory", "amber", "velvet", "cotton", "linen", "marble",
    "copper", "bronze", "crystal", "leather", "rubber", "plastic",
]


def build_op6():
    rows = []
    words = [p[0] for p in OP7_PAIRS]
    all_corrupts = words + CORRUPT_WORDS_POOL
    for word, target, foil, _ in OP7_PAIRS:
        for clean_tpl, corrupt_tpl in OP7_TEMPLATES:
            clean = clean_tpl.format(word=word)
            cl = tok_len(clean)
            used = set()
            for corrupt_word in all_corrupts:
                if corrupt_word == word:
                    continue
                corrupted = corrupt_tpl.format(corrupt_word=corrupt_word)
                if tok_len(corrupted) != cl:
                    continue
                if corrupt_word in used:
                    continue
                row = audit_pair(clean, corrupted, target, foil)
                if row:
                    rows.append(row)
                    used.add(corrupt_word)
                if len(used) >= 4:
                    break
    return rows


# ============================================================
# Main
# ============================================================
BUILDERS = {
    "op1_hypocorism": build_op1,
    "op2_clipping": build_op2,
    "op3_initialism": build_op3,
    "op4_oronym": build_op4,
    "op5_homophone": build_op5,
    "op6_folk_etym": build_op6,
}


def run_audit():
    """Print tokenization audit for all raw pairs before filtering."""
    print("\n=== TOKENIZATION AUDIT ===\n")

    print("--- Op 3: Letter names ---")
    for letter, (name, foil) in OP3_MAP.items():
        tid = tok_id(name)
        fid = tok_id(foil)
        status = "OK" if tid is not None else "MULTI-TOKEN"
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        print(f"  {letter} -> {name:8s}  id={tid}  tokens={toks}  {status}")

    print("\n--- Op 4: Oronym targets ---")
    for first, last, target, foil, corrupt in OP4_PAIRS:
        tid = tok_id(target)
        toks = tokenizer.encode(" " + target, add_special_tokens=False)
        clean = f"{first} {last} sounds like"
        corrupted = f"{corrupt} {last} sounds like"
        cl = tok_len(clean)
        cr = tok_len(corrupted)
        len_match = "OK" if cl == cr else f"MISMATCH ({cl} vs {cr})"
        tok_status = "OK" if tid is not None else "MULTI-TOKEN"
        print(f"  {first:8s} {last:10s} -> {target:15s}  id={tid}  tokens={toks}  {tok_status}  len={len_match}")

    print("\n--- Op 1: Nickname targets ---")
    for name, nickname, foil in OP1_PAIRS[:10]:
        tid = tok_id(nickname)
        toks = tokenizer.encode(" " + nickname, add_special_tokens=False)
        status = "OK" if tid is not None else "MULTI-TOKEN"
        print(f"  {name:12s} -> {nickname:6s}  id={tid}  tokens={toks}  {status}")
    if len(OP1_PAIRS) > 10:
        print(f"  ... and {len(OP1_PAIRS) - 10} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build phonetic circuit datasets")
    parser.add_argument("--ops", nargs="+", default=list(BUILDERS.keys()),
                        help="Which operations to build")
    parser.add_argument("--audit-only", action="store_true",
                        help="Only run tokenization audit, don't write CSVs")
    parser.add_argument("--output-dir", type=str, default=str(DATASETS_DIR))
    args = parser.parse_args()

    run_audit()

    if args.audit_only:
        print("\nAudit only — no CSVs written.")
        raise SystemExit(0)

    print("\n=== BUILDING DATASETS ===\n")
    out_dir = Path(args.output_dir)
    summary = {}
    for op in args.ops:
        if op not in BUILDERS:
            print(f"  SKIP unknown op: {op}")
            continue
        rows = BUILDERS[op]()
        write_csv(rows, out_dir / f"{op}.csv", op)
        summary[op] = len(rows)

    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    total = sum(summary.values())
    print(f"Total pairs: {total}")
