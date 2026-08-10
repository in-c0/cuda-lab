"""Summarise the SASS instruction mix of each kernel variant.

  cuobjdump --dump-sass cppn.exe > sass.txt
  python sass_summary.py            # writes sass_summary.txt
"""
import re
from collections import Counter

names = {'Li0ELi0ELi0E': 'A  global   + no early-out',
         'Li1ELi0ELi0E': '   constant + no early-out',
         'Li0ELi1ELi0E': '   global   + early-out',
         'Li1ELi1ELi0E': 'B  constant + early-out',
         'Li1ELi1ELi1E': 'C  constant + early-out + intrinsics',
         'Li0ELi1ELi1E': 'D  global   + early-out + intrinsics'}

txt = open('sass.txt', errors='ignore').read()
parts = re.split(r'\n\s*Function : ', txt)
out = ["SASS instruction mix -- cuobjdump --dump-sass cppn.exe, sm_75",
       "NOTE: 'instrs' is a LOWER BOUND for any kernel reported at 4096: the",
       "disassembly offsets are 4 hex digits and wrap at 65536 bytes = 4096",
       "instructions. Only the intrinsic variants are short enough to count exactly.",
       "",
       "%-40s %7s %5s %5s %5s %6s  %s" % ('kernel', 'instrs', 'LDG', 'LDC', 'MUFU', 'FFMA', 'MUFU kinds')]
for p in parts[1:]:
    mangled = p.split('\n')[0].strip()
    key = [k for k in names if k in mangled]
    if not key:
        continue
    ops = [m[1] for m in re.findall(r'/\*[0-9a-f]{4}\*/\s+(@?!?P?\d?\s*)?([A-Z][A-Z0-9._@]*)', p)]
    c = Counter(ops)
    ldg = sum(v for k, v in c.items() if k.startswith('LDG'))
    ldc = sum(v for k, v in c.items() if k.startswith('LDC'))
    mufu = sum(v for k, v in c.items() if k.startswith('MUFU'))
    kinds = ' '.join(sorted(k for k in c if k.startswith('MUFU')))
    out.append("%-40s %7d %5d %5d %5d %6d  %s" %
               (names[key[0]], len(ops), ldg, ldc, mufu, c.get('FFMA', 0), kinds))
open('sass_summary.txt', 'w').write("\n".join(out) + "\n")
print("\n".join(out))
