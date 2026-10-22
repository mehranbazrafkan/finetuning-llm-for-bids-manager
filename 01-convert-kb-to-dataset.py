"""
01-convert-kb-to-dataset.py

Converts the BIDS knowledge base (knowledge.jsonl) into a prompt/completion
fine-tuning dataset that can be used directly with 02-LoRA-fine-tuning.py.

The knowledge base is a RETRIEVAL corpus, not training pairs. This script
synthesizes RAG-style training samples where:

    prompt     = [retrieved knowledge block] + user question
    completion = a helpful, accurate answer built from that knowledge

This mirrors how the agent will work at inference time (retrieve KB entries,
inject them into the prompt, let the LLM answer), which is the only realistic
way for a 270M-parameter model to use 2,488 knowledge records.

data/knowledge.jsonl is the single consolidated knowledge file (base knowledge +
enriched records merged and deduplicated, 2488 records). Records that carry AI
enrichment expose it inline via their "ai_enrichment" field (description,
interpretation, common_causes, resolution_guidance, ...), which this script uses
to produce richer support-style answers. No second input file is needed.

Usage:
    python 01-convert-kb-to-dataset.py
    python 01-convert-kb-to-dataset.py --limit-per-type 200   # quick test run

Input is read from data/knowledge.jsonl, output is written to data/.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ==================================================
# Helpers
# ==================================================

def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping bad JSON on line {line_no} of {path.name}: {e}")
    return records


def clean_ws(text) -> str:
    """Collapse newlines / repeated spaces into a single space."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def clip(text: str, limit: int) -> str:
    """Trim text to ~limit chars, preferring a sentence boundary."""
    text = clean_ws(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "; ", " "):
        idx = cut.rfind(sep)
        if idx > int(limit * 0.6):
            return cut[: idx + 1].rstrip()
    return cut.rstrip() + "..."


def join_items(items, sep="; ") -> str:
    return sep.join(clean_ws(str(i)) for i in items if str(i).strip())


# ==================================================
# $ref resolution (templates.yaml and file rules reference each other)
# ==================================================

class RefResolver:
    """
    Expands $ref chains inside raw_content structures.

    Ref forms seen in the knowledge base:
        meta.templates.deriv.base                      (template: file + section + key)
        rules.files.raw.beh.noncontinuous              (file rule: file + key)
        meta.templates.deriv.base.entities             (sub-key of the record above)
    """

    def __init__(self, kb: list):
        self.index = {}
        for rec in kb:
            src = rec.get("source") or {}
            f, key = src.get("file"), src.get("key")
            if not f or key is None:
                continue
            base = f[:-5] if f.endswith(".yaml") else f
            base = base.replace("/", ".")
            self.index[base + "." + str(key)] = rec
            section = src.get("section")
            if section:
                self.index[base + "." + str(section) + "." + str(key)] = rec

    def _lookup(self, ref: str, seen: set):
        """Resolve a ref string to a node inside some record's raw_content."""
        if ref in seen:
            return None
        parts = str(ref).split(".")
        for cut in range(len(parts)):
            cand = ".".join(parts[: len(parts) - cut])
            rec = self.index.get(cand)
            if rec is None:
                continue
            node = rec.get("raw_content")
            for sub in parts[len(parts) - cut:]:
                if isinstance(node, dict) and sub in node:
                    node = node[sub]
                else:
                    return None
            return node
        return None

    def deep(self, node, seen=None):
        """Recursively expand every $ref inside a raw_content structure."""
        seen = seen or set()
        if isinstance(node, dict):
            if "$ref" in node:
                refs = node["$ref"]
                if isinstance(refs, str):
                    refs = [refs]
                base = {}
                for rf in refs:
                    target = self._lookup(rf, seen)
                    if isinstance(target, dict):
                        base.update(self.deep(target, seen | {rf}))
                rest = {k: v for k, v in node.items() if k != "$ref"}
                base.update(self.deep(rest, seen))
                return base
            return {k: self.deep(v, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [self.deep(v, seen) for v in node]
        return node


def format_entities(entities) -> str:
    if not isinstance(entities, dict) or not entities:
        return ""
    parts = []
    for name, rule in entities.items():
        rule_text = rule if isinstance(rule, str) else json.dumps(rule)
        parts.append(f"{name} ({rule_text})")
    return ", ".join(parts)


# ==================================================
# Enum values ($ref entries leak as 'objects.enums.CapTrak.value')
# ==================================================

def enum_values(rec: dict) -> list:
    raw = rec.get("allowed_values") or (rec.get("raw_content") or {}).get("enum") or []
    values = []
    for v in raw:
        if isinstance(v, str):
            values.append(v)
        elif isinstance(v, dict) and "$ref" in v:
            parts = str(v["$ref"]).split(".")
            if len(parts) >= 2:
                values.append(parts[-2])  # objects.enums.CapTrak.value -> CapTrak
    return list(dict.fromkeys(values))


# ==================================================
# Context block (what the agent will inject at inference time)
# ==================================================

EXTRA_FACT_TYPES = {"MetadataRule", "TabularRule", "Check", "Association"}


def build_context(rec: dict, enrich: dict, resolver: RefResolver) -> str:
    kt = rec["knowledge_type"]
    src = rec.get("source") or {}
    lines = [f"Title: {rec['title']}"]

    summary = clean_ws(rec.get("summary"))
    if summary:
        lines.append(f"Summary: {summary}")

    # --- facts line (per type; retrieval_text is unreliable for some types) ---
    if kt == "Enum":
        rc = rec.get("raw_content") or {}
        et = (rec.get("scope") or {}).get("enum_type") or rc.get("type")
        facts = ""
        if rc.get("value") is not None and "enum" not in rc:
            # single named value, e.g. {value: "L", display_name: "Left Hemisphere", description: ...}
            facts = f"Value: {rc['value']}."
            if rc.get("display_name"):
                facts += f" Display name: {rc['display_name']}."
        else:
            values = enum_values(rec)
            if values:
                facts = f"Allowed values: {', '.join(values)}."
                if et:
                    facts = f"Value type: {et}. " + facts
    elif kt == "Template":
        raw = resolver.deep(rec.get("raw_content") or {})
        facts = ""
        entities = format_entities(raw.get("entities"))
        if entities:
            facts = f"Entities: {entities}."
        if raw.get("extensions"):
            facts += f" Allowed extensions: {', '.join(map(str, raw['extensions']))}."
        if raw.get("suffixes"):
            facts += f" Suffixes: {', '.join(map(str, raw['suffixes']))}."
        facts = facts.strip()
    elif kt == "FileSpecification" and src.get("section") == "rules":
        # datatype file rules: build facts from resolved fields (retrieval_text leaks $refs)
        scope = rec.get("scope") or {}
        raw = resolver.deep(rec.get("raw_content") or {})
        facts = ""
        entities = format_entities(raw.get("entities"))
        if entities:
            facts += f"Entities: {entities}."
        if scope.get("suffixes"):
            facts += f" Suffixes: {', '.join(map(str, scope['suffixes']))}."
        if scope.get("extensions"):
            facts += f" Extensions: {', '.join(map(str, scope['extensions']))}."
        if scope.get("datatypes"):
            facts += f" Applies to datatypes: {', '.join(map(str, scope['datatypes']))}."
        else:
            facts += " Applies to any datatype."
        if raw.get("level"):
            facts += f" The file itself is {raw['level']}."
        facts = facts.strip()
    else:
        facts = clip(rec.get("retrieval_text"), 650)

    if facts:
        lines.append(f"Facts: {facts}")

    # --- exact rule syntax (helpful for matching real validator messages) ---
    if kt in EXTRA_FACT_TYPES:
        conds = rec.get("conditions") or (rec.get("raw_content") or {}).get("selectors")
        if conds:
            lines.append(f"Applies when: {join_items(conds)}")
        req = rec.get("requirements")
        if isinstance(req, dict):
            req_bits = [
                f"{k}={join_items(v, ', ') if isinstance(v, list) else v}"
                for k, v in req.items()
                if v is not None and k not in ("message", "issue_code")
            ]
            if req_bits:
                lines.append(f"Rule requires: {', '.join(req_bits)}")

    # --- AI enrichment (only exists for the enriched subset) ---
    if enrich.get("description"):
        lines.append(f"Explanation: {clip(enrich['description'], 350)}")

    return "\n".join(lines)


def make_prompt(context: str, question: str) -> str:
    if context:
        return f"Retrieved BIDS knowledge:\n{context}\n\nQuestion: {question}"
    return question


# ==================================================
# QA generation per knowledge type
# ==================================================

def issue_parts(rec: dict):
    """Pull (code, message, level, selectors, checks) out of Error/Warning/Check records."""
    rc = rec.get("raw_content") or {}
    issue = rc.get("issue") if isinstance(rc.get("issue"), dict) else rc
    scope = rec.get("scope") or {}
    req = rec.get("requirements") if isinstance(rec.get("requirements"), dict) else {}

    code = issue.get("code") or scope.get("issue_code") or req.get("issue_code") or req.get("code") or rec.get("id")
    message = clean_ws(issue.get("message") or rec.get("summary") or "")
    level = (issue.get("level") or rec.get("severity") or scope.get("severity") or "error").lower()
    selectors = rc.get("selectors") or rec.get("conditions") or []
    checks = rc.get("checks") or []
    return code, message, level, selectors, checks


def qa_for_record(rec: dict, enrich: dict, resolver: RefResolver) -> list:
    """Return a list of (context, question, answer) triples for one KB record."""
    kt = rec["knowledge_type"]
    context = build_context(rec, enrich, resolver)
    out = []

    # --------------------------------------------------
    # Error / Warning / Check  (what users actually see)
    # --------------------------------------------------
    if kt in ("Error", "Warning", "Check"):
        code, message, level, selectors, checks = issue_parts(rec)
        impact = (
            "This makes the dataset invalid."
            if level == "error"
            else "The dataset is still valid, but the issue should be reviewed."
        )

        a1 = f"`{code}` is a BIDS validation {level.upper()}. {message} {impact}"
        if selectors:
            a1 += f" It is triggered when: {join_items(selectors)}."
        if checks and not enrich:
            a1 += f" The validator then verifies: {join_items(checks)}."
        if enrich.get("description"):
            a1 += f" {clip(enrich['description'], 300)}"
        if enrich.get("interpretation"):
            a1 += f" {clip(enrich['interpretation'], 300)}"

        out.append((
            context,
            f"I'm getting the BIDS validation {level.upper()} {code} in BIDS-Manager. What does it mean?",
            a1.strip(),
        ))

        guidance = enrich.get("resolution_guidance")
        causes = enrich.get("common_causes") or []
        if guidance or causes:
            a2 = ""
            if causes:
                a2 += f"Common causes: {join_items([c.rstrip('. ') for c in causes], '; ')}. "
            if guidance:
                a2 += clean_ws(guidance)
            out.append((
                context,
                f"How do I resolve the {code} {level}?",
                a2.strip(),
            ))

    # --------------------------------------------------
    # MetadataRule (JSON sidecar fields)
    # --------------------------------------------------
    elif kt == "MetadataRule":
        scope = rec.get("scope") or {}
        req = rec.get("requirements") if isinstance(rec.get("requirements"), dict) else {}
        field = req.get("field") or join_items(scope.get("metadata_fields"), ", ") or rec["source"]["key"]
        out.append((
            context,
            f"What are the rules for the '{field}' JSON sidecar metadata field in BIDS?",
            clip(rec.get("retrieval_text"), 700),
        ))

    # --------------------------------------------------
    # TabularRule (TSV columns)
    # --------------------------------------------------
    elif kt == "TabularRule":
        scope = rec.get("scope") or {}
        req = rec.get("requirements") if isinstance(rec.get("requirements"), dict) else {}
        column = req.get("column") or join_items(scope.get("columns"), ", ") or rec["source"]["key"]
        table = join_items(scope.get("tabular_files"), ", ") or "this BIDS"
        out.append((
            context,
            f"Is the '{column}' column required in the {table} TSV table?",
            clip(rec.get("retrieval_text"), 700),
        ))

    # --------------------------------------------------
    # Concept (context namespaces, BIDS entities, ...)
    # --------------------------------------------------
    elif kt == "Concept":
        rc = rec.get("raw_content") or {}
        if rc.get("context_key"):
            name = rc["context_key"]
            question = f"What is the '{name}' context namespace in BIDS validation?"
        elif rc.get("name") and rc.get("description"):
            name = rc["name"]
            display = rc.get("display_name") or name
            question = f"What is the '{name}' ({display}) entity in BIDS filenames?"
        else:
            name = rc.get("display_name") or rec["source"]["key"]
            question = f"What is '{name}' in BIDS?"

        answer = clip(rc.get("description") or rec.get("retrieval_text"), 700)
        if enrich.get("interpretation"):
            answer += f" {clip(enrich['interpretation'], 300)}"
        out.append((context, question, answer.strip()))

    # --------------------------------------------------
    # Definition (common principles)
    # --------------------------------------------------
    elif kt == "Definition":
        rc = rec.get("raw_content") or {}
        name = rc.get("display_name") or rec["source"]["key"]
        answer = clip(rc.get("description") or rec.get("retrieval_text"), 800)
        if enrich.get("interpretation"):
            answer += f" {clip(enrich['interpretation'], 250)}"
        out.append((context, f"What is {name} in BIDS?", answer.strip()))

    # --------------------------------------------------
    # Enum (value lists, or single named values like 'L' = Left Hemisphere)
    # --------------------------------------------------
    elif kt == "Enum":
        rc = rec.get("raw_content") or {}
        key = rec["source"]["key"]
        if rc.get("value") is not None and "enum" not in rc:
            display = rc.get("display_name") or key
            a = f"In BIDS, '{key}' ({display}) has the value: {rc['value']}."
            desc = clean_ws(rc.get("description"))
            if desc:
                a += f" {clip(desc, 400)}"
            if rc.get("tags"):
                a += f" It is used with: {join_items(rc['tags'], ', ')}."
            out.append((context, f"What is the '{key}' ({display}) enumeration value in BIDS?", a.strip()))
        else:
            values = enum_values(rec)
            if not values:
                return []
            et = (rec.get("scope") or {}).get("enum_type") or rc.get("type")
            a = f"The allowed values for '{key}' are: {', '.join(values)}."
            if et:
                a += f" The value type is {et}."
            out.append((context, f"What values are allowed for '{key}' in BIDS?", a))

    # --------------------------------------------------
    # FileSpecification (top-level dataset files, or datatype file rules)
    # --------------------------------------------------
    elif kt == "FileSpecification":
        src = rec.get("source") or {}
        rc = rec.get("raw_content") or {}
        if src.get("section") == "rules":
            scope = rec.get("scope") or {}
            raw = resolver.deep(rc)
            name = src["key"]
            entities = format_entities(raw.get("entities"))
            a = f"In BIDS, a '{name}' file must follow these filename rules:"
            if entities:
                a += f" entities: {entities}."
            if scope.get("suffixes"):
                a += f" The filename uses the suffix: {', '.join(map(str, scope['suffixes']))}."
            if scope.get("extensions"):
                a += f" Allowed extensions: {', '.join(map(str, scope['extensions']))}."
            if scope.get("datatypes"):
                a += f" It applies to the datatypes: {', '.join(map(str, scope['datatypes']))}."
            else:
                a += " It can appear in any datatype directory."
            if raw.get("level"):
                a += f" The file itself is {raw['level']}."
            out.append((
                context,
                f"What are the filename requirements for a '{name}' file in BIDS?",
                a.strip(),
            ))
        else:
            name = src["key"]
            answer = clip(rc.get("description") or rec.get("retrieval_text"), 600)
            if rc.get("file_type"):
                answer += f" (file type: {rc['file_type']})"
            out.append((context, f"What is the '{name}' file in a BIDS dataset?", answer.strip()))

    # --------------------------------------------------
    # DirectoryRule (dataset layout)
    # --------------------------------------------------
    elif kt == "DirectoryRule":
        dtype = (rec.get("scope") or {}).get("dataset_type") or rec["source"]["key"]
        out.append((
            context,
            f"What root-level directories does a {dtype} BIDS dataset define?",
            clip(rec.get("retrieval_text"), 500),
        ))

    # --------------------------------------------------
    # Relationship (modality -> datatypes)
    # --------------------------------------------------
    elif kt == "Relationship":
        scope = rec.get("scope") or {}
        modality = join_items(scope.get("modality"), ", ") or rec["source"]["key"]
        out.append((
            context,
            f"Which datatypes belong to the '{modality}' BIDS modality?",
            clip(rec.get("retrieval_text"), 400),
        ))

    # --------------------------------------------------
    # Template (filename entity templates; has $ref chains)
    # --------------------------------------------------
    elif kt == "Template":
        raw = resolver.deep(rec.get("raw_content") or {})
        tmpl = (rec.get("scope") or {}).get("template") or rec["source"]["key"]
        section = rec["source"].get("section") or "file"
        entities = format_entities(raw.get("entities"))
        if not entities:
            return []
        a = f"The '{tmpl}' {section} BIDS file template uses these filename entities: {entities}."
        if raw.get("extensions"):
            a += f" Allowed extensions: {', '.join(map(str, raw['extensions']))}."
        if raw.get("suffixes"):
            a += f" Allowed suffixes: {', '.join(map(str, raw['suffixes']))}."
        out.append((
            context,
            f"Which filename entities does the '{tmpl}' {section} BIDS template use?",
            a,
        ))

    # --------------------------------------------------
    # Association (associated-file lookup rules)
    # --------------------------------------------------
    elif kt == "Association":
        rc = rec.get("raw_content") or {}
        target = rc.get("target") or {}
        name = rec["source"]["key"]
        suffix, ext = target.get("suffix"), target.get("extension")
        if suffix and ext:
            target_desc = f"an associated '{suffix}{ext}' file"
        elif ext:
            target_desc = f"an associated file with extension '{ext}'"
        elif suffix:
            target_desc = f"an associated '{suffix}' file"
        else:
            target_desc = "an associated file"
        selectors = rc.get("selectors") or rec.get("conditions") or []
        inherit = "may be inherited from a parent directory" if rc.get("inherit") else "must be in the same directory as the data file"

        a1 = f"BIDS-Manager looks for {target_desc}"
        if selectors:
            a1 += f" when: {join_items(selectors)}."
        a1 += f" The associated file {inherit}."
        if enrich.get("interpretation"):
            a1 += f" {clip(enrich['interpretation'], 250)}"
        out.append((context, f"When does BIDS-Manager look for an associated '{name}' file?", a1.strip()))

    # --------------------------------------------------
    # Version
    # --------------------------------------------------
    elif kt == "Version":
        bv = rec.get("bids_version") or (rec.get("raw_content") or {}).get("version", "?")
        sv = rec.get("schema_version") or (rec.get("raw_content") or {}).get("schema_version")
        a = f"This knowledge base targets BIDS specification version {bv}"
        if sv:
            a += f" and BIDS schema version {sv}"
        out.append((context, "Which BIDS and schema versions does this knowledge base target?", a + "."))

    return out


# ==================================================
# Global "persona" pairs (no retrieval context)
# ==================================================

def global_pairs() -> list:
    none = None
    return [
        (
            none,
            "What can you help me with?",
            "I can help you work with BIDS-Manager and the Brain Imaging Data Structure (BIDS) standard. "
            "I can explain validation errors and warnings, JSON sidecar metadata field rules, TSV table column "
            "requirements, filename entities and templates, dataset directory structure, and how to fix "
            "validation problems in your dataset. Just paste the error message or tell me what you are trying to do.",
        ),
        (
            none,
            "My BIDS dataset fails validation in BIDS-Manager. Where should I start?",
            "Start by reading the validation report: fix every ERROR first, because errors make the dataset invalid, "
            "then review WARNINGs, which keep the dataset valid but indicate non-ideal structure. For each code in the "
            "report, look up what triggers it and which files or metadata fields it concerns, then apply the fix and "
            "re-validate.",
        ),
        (
            none,
            "What is BIDS?",
            "BIDS (Brain Imaging Data Structure) is a standard for organizing neuroimaging and behavioral datasets. "
            "It prescribes a directory layout, filename entities such as subject, session, task and run, JSON sidecar "
            "metadata, and TSV tables, so that datasets are readable by humans, tools, and validators alike.",
        ),
        (
            none,
            "What is the difference between an error and a warning in BIDS validation?",
            "An ERROR means the dataset violates the BIDS specification and is considered invalid; it must be fixed. "
            "A WARNING means the dataset is still valid BIDS, but something is unusual or discouraged; you should "
            "review it and fix it when it reflects a real problem.",
        ),
    ]


# ==================================================
# Main
# ==================================================

def main():
    parser = argparse.ArgumentParser(description="Convert the BIDS knowledge base into a fine-tuning dataset.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "knowledge.jsonl",
                        help="Consolidated knowledge file (knowledge + enrichment merged).")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "knowledge-fine-tune-dataset.jsonl")
    parser.add_argument("--limit-per-type", type=int, default=0, help="Cap samples per knowledge_type (0 = no cap).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading knowledge base: {args.input}")
    kb = load_jsonl(args.input)
    print(f"  {len(kb)} records, types: {dict(Counter(r['knowledge_type'] for r in kb))}")

    # Enrichment rides inline on each record ("ai_enrichment" field), so a
    # separate enrichment file is no longer needed.
    enrich_by_id = {r["id"]: r["ai_enrichment"] for r in kb if r.get("ai_enrichment")}
    print(f"Records with inline ai_enrichment: {len(enrich_by_id)}")

    resolver = RefResolver(kb)

    # --------------------------------------------------
    # Generate QA pairs
    # --------------------------------------------------
    samples = []          # (type, context, question, answer)
    per_type = Counter()

    for rec in kb:
        kt = rec["knowledge_type"]
        if args.limit_per_type and per_type[kt] >= args.limit_per_type:
            continue
        enrich = enrich_by_id.get(rec["id"], {})
        for context, question, answer in qa_for_record(rec, enrich, resolver):
            if not answer or not question:
                continue
            samples.append((kt, context, question, answer))
            per_type[kt] += 1

    # --------------------------------------------------
    # Add persona pairs
    # --------------------------------------------------
    for context, question, answer in global_pairs():
        samples.append(("Global", context, question, answer))

    # --------------------------------------------------
    # Dedupe by question, shuffle, write
    # --------------------------------------------------
    seen_questions = set()
    unique_samples = []
    for s in samples:
        key = clean_ws(s[2]).lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        unique_samples.append(s)

    rng = random.Random(args.seed)
    rng.shuffle(unique_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for _kt, context, question, answer in unique_samples:
            record = {"prompt": make_prompt(context, question), "completion": answer}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --------------------------------------------------
    # Stats
    # --------------------------------------------------
    n = len(unique_samples)
    prompt_lens = [len(s[2]) + len(s[3] or "") + 80 for s in unique_samples]  # +80 ~ context wrapper
    completion_lens = [len(s[3]) for s in unique_samples]
    est_tokens = [(p + c) // 4 for p, c in zip(prompt_lens, completion_lens)]

    print("\n-------------------- stats --------------------")
    print(f"Total training samples : {n}")
    print("Samples per type       :")
    for kt, count in sorted(per_type.items(), key=lambda x: -x[1]):
        print(f"    {kt:<16} {count}")
    print(f"Avg prompt chars       : {sum(prompt_lens) // n}")
    print(f"Max prompt chars       : {max(prompt_lens)}  (~{max(est_tokens)} tokens)")
    print(f"Avg completion chars   : {sum(completion_lens) // n}")
    print(f"Max completion chars   : {max(completion_lens)}")
    print(f"Output                 : {args.output}")
    print("------------------------------------------------\n")

    # --------------------------------------------------
    # Show two samples
    # --------------------------------------------------
    for kt, context, question, answer in unique_samples[:2]:
        print(f"--- sample ({kt}) ---")
        print("PROMPT:")
        print(make_prompt(context, question)[:600])
        print("\nCOMPLETION:")
        print(answer[:600])
        print()


if __name__ == "__main__":
    main()
