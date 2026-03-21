"""Cross-check LaTeX \cite{} keys against \bibitem{} definitions."""
import re
import sys

with open("paper.tex", "r", encoding="utf-8") as f:
    content = f.read()

# Extract all \cite keys (handles \cite{a,b,c} multi-cites)
cite_pattern = re.compile(r"\\cite\{([^}]+)\}")
cite_matches = cite_pattern.findall(content)
cited_keys = set()
for match in cite_matches:
    for key in match.split(","):
        cited_keys.add(key.strip())

# Extract all \bibitem keys
bibitem_pattern = re.compile(r"\\bibitem\{([^}]+)\}")
bibitem_keys = set(bibitem_pattern.findall(content))

# Cross-check
cited_not_defined = cited_keys - bibitem_keys
defined_not_cited = bibitem_keys - cited_keys

print(f"=== CITED KEYS ({len(cited_keys)}) ===")
for k in sorted(cited_keys):
    status = "OK" if k in bibitem_keys else "MISSING BIBITEM"
    print(f"  {k}: {status}")

print(f"\n=== DEFINED BUT NEVER CITED ({len(defined_not_cited)}) ===")
for k in sorted(defined_not_cited):
    print(f"  {k}")

print()
if cited_not_defined:
    print(f"!!! CITED BUT NO BIBITEM ({len(cited_not_defined)}) !!!")
    for k in sorted(cited_not_defined):
        print(f"  {k}")
    sys.exit(1)
else:
    print("All cited references have matching bibitems.")

# Also check for figure references
fig_ref_pattern = re.compile(r"\\ref\{([^}]+)\}")
fig_refs = set(fig_ref_pattern.findall(content))
label_pattern = re.compile(r"\\label\{([^}]+)\}")
labels = set(label_pattern.findall(content))

ref_not_labeled = fig_refs - labels
label_not_reffed = labels - fig_refs

print(f"\n=== CROSS-REFERENCES ({len(fig_refs)}) ===")
for r in sorted(fig_refs):
    status = "OK" if r in labels else "MISSING LABEL"
    print(f"  {r}: {status}")

print(f"\n=== LABELS NEVER REFERENCED ({len(label_not_reffed)}) ===")
for l in sorted(label_not_reffed):
    print(f"  {l}")

if ref_not_labeled:
    print(f"\n!!! REF WITHOUT LABEL ({len(ref_not_labeled)}) !!!")
    for r in sorted(ref_not_labeled):
        print(f"  {r}")

# Check figure files exist
import os
fig_pattern = re.compile(r"\\includegraphics[^{]*\{([^}]+)\}")
fig_files = fig_pattern.findall(content)
print(f"\n=== FIGURE FILES ({len(fig_files)}) ===")
for f in fig_files:
    exists = os.path.exists(f)
    print(f"  {f}: {'OK' if exists else 'FILE NOT FOUND'}")
