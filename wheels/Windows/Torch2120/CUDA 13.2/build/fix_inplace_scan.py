# Rewrites in-place cub::DeviceScan::{Exclusive,Inclusive}Sum(temp, bytes, data, n) calls
# to the explicit (temp, bytes, data, data, n) form, which CCCL 3.x no longer confuses
# with the new (in, out, num_items, env) overload.
import re, sys, pathlib

pat = re.compile(r'(cub::DeviceScan::(?:Exclusive|Inclusive)Sum\()([^;]*?)(\)\s*\)?\s*;)', re.S)


def split_args(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def repl(m):
    args = split_args(m.group(2))
    if len(args) != 4:
        return m.group(0)
    args.insert(3, ' ' + args[2].strip())
    return m.group(1) + ','.join(args) + m.group(3)


for root in sys.argv[1:]:
    for p in pathlib.Path(root).rglob('*'):
        if p.suffix not in ('.cu', '.cuh', '.h', '.hpp', '.cpp'):
            continue
        src = p.read_bytes().decode('utf-8')
        new, n = pat.subn(repl, src)
        if new != src:
            p.write_bytes(new.encode('utf-8'))
            print(p)
