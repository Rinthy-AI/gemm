#!/usr/bin/env python3

"""
Read a markdown document from stdin, extract all of the cpp code blocks,
concatenate them, and write them to stdout
"""

import sys

in_code_block = False
for line_unstripped in sys.stdin.readlines():
    line = line_unstripped.rstrip()
    if line == "```cpp" and not in_code_block:
        in_code_block = True
        continue
    elif line == "```" and in_code_block:
        in_code_block = False
        continue
    if in_code_block:
        print(line)
