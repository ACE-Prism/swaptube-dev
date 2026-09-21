#!/usr/bin/env python3
"""Generate src/CPU/ from src/CUDA/ for the CPU backend.

This mirrors what the HIP path does with hipify-perl: src/CUDA/ stays the single
source of truth and the backend-specific tree is generated at build time. Do not
edit src/CPU/ by hand.

The only thing that cannot be handled by macros in cuda_runtime.h is the
<<<grid, block>>> launch syntax, which no C++ compiler can parse. Every launch
in this codebase has the form

    kernel<<<grid, block>>>(args...)

so substituting just the <<<...>>> token is enough:

    swaptube_cpu::launcher("kernel", kernel, grid, block)(args...)

The argument list is never touched, which keeps the rewrite independent of how
the arguments are formatted or nested. The name is passed through only so that
SWAPTUBE_PROFILE_KERNELS=1 can attribute time to individual kernels.
"""

import argparse
import pathlib
import re
import shutil
import sys

# Matches `name<<<a, b>>>`. The payloads in this codebase never contain '>',
# which is what makes the non-greedy character class safe.
LAUNCH = re.compile(r'\b(?P<kernel>[A-Za-z_]\w*)\s*<<<(?P<config>[^<>]*)>>>')

KERNEL_DEF = re.compile(r'__global__\s+\w[\w:<>,\s*&]*?\b(?P<name>\w+)\s*\(')


def kernel_body(src: str, open_brace: int) -> str:
    """Return the braced body starting at open_brace, or '' if unbalanced."""
    depth = 0
    for i in range(open_brace, len(src)):
        c = src[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[open_brace:i + 1]
    return ''


def classify_kernels(sources):
    """Split __global__ kernels into those needing real barriers and the rest.

    A kernel needs the barrier dispatcher only if it communicates between the
    threads of a block through __shared__ memory. Threads within a block run
    sequentially on the fast path, so a kernel whose __syncthreads only orders
    global atomics still produces the right answer -- and running it on the fast
    path matters, because reverse_conway_kernel is launched with 65536 blocks of
    256 threads and would otherwise need 256 OS threads and 16M barrier waits.
    """
    barrier, atomics_only = set(), set()
    for path in sources:
        src = path.read_text()
        for m in KERNEL_DEF.finditer(src):
            brace = src.find('{', m.end() - 1)
            if brace < 0:
                continue
            body = kernel_body(src, brace)
            if '__syncthreads' not in body:
                continue
            if '__shared__' in body:
                barrier.add(m.group('name'))
            else:
                atomics_only.add(m.group('name'))
    return barrier, atomics_only


def include_line(rel: pathlib.Path) -> str:
    # src/CPU/foo.cu -> ../CPUBackend, src/CPU/Beavers/bar.cu -> ../../CPUBackend
    up = '../' * len(rel.parts)
    return f'#include "{up}CPUBackend/LaunchCPU.h"\n'


# `extern "C" <return type> <name>(`, where the name is what the C++ layer calls.
EXTERN_C_ENTRY = re.compile(r'(extern\s+"C"\s+[\w:*&<>\s]*?\b)(?P<name>\w+)(\s*\()')


def suffix_extern_c_entries(src: str) -> tuple:
    """Rename this file's extern "C" entry points with a _cpu suffix.

    Applied to files that have a Metal port. Both implementations then link
    together, and the Metal wrapper in src/Metal/ owns the real name and forwards
    to the _cpu one when the GPU is unavailable or SWAPTUBE_DISABLE_METAL is set.
    That fallback is also what makes an A/B comparison possible from a single
    build.
    """
    renamed = []

    def sub(m):
        renamed.append(m.group('name'))
        return f'{m.group(1)}{m.group("name")}_cpu{m.group(3)}'

    return EXTERN_C_ENTRY.sub(sub, src), renamed


def translate(src: str, rel: pathlib.Path, barrier_kernels, metal_ported=False) -> tuple:
    launches = 0
    barrier_launches = 0
    renamed = []

    if metal_ported:
        src, renamed = suffix_extern_c_entries(src)

    def sub(m):
        nonlocal launches, barrier_launches
        launches += 1
        kernel = m.group('kernel')
        config = m.group('config').strip()
        fn = 'launcher'
        if kernel in barrier_kernels:
            fn = 'launcher_barrier'
            barrier_launches += 1
        return f'swaptube_cpu::{fn}("{kernel}", {kernel}, {config})'

    out = LAUNCH.sub(sub, src)

    # The shim must be in scope before anything else, so the qualifier macros
    # are defined ahead of the file's own includes.
    header = include_line(rel)
    if out.lstrip().startswith('#pragma once'):
        idx = out.index('#pragma once') + len('#pragma once')
        out = out[:idx] + '\n' + header + out[idx:].lstrip('\n')
    else:
        out = header + out

    return out, launches, barrier_launches, renamed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=pathlib.Path)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    ap.add_argument('--metal-manifest', type=pathlib.Path,
                    help='files listed here get their extern "C" entry points suffixed '
                         'with _cpu, because src/Metal owns the real names')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    metal_ported = set()
    if args.metal_manifest and args.metal_manifest.exists():
        for line in args.metal_manifest.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                metal_ported.add(line.split()[0])

    in_root, out_root = args.input, args.output
    if not in_root.is_dir():
        print(f'translate_cuda_to_cpu: no such directory: {in_root}', file=sys.stderr)
        return 1

    sources = sorted(list(in_root.rglob('*.cu')) + list(in_root.rglob('*.cuh')))
    barrier_kernels, atomics_only = classify_kernels(sources)

    if out_root.exists():
        shutil.rmtree(out_root)

    total_launches = total_barrier = 0
    all_renamed = {}
    for path in sources:
        rel = path.relative_to(in_root)
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        out, n, nb, renamed = translate(path.read_text(), rel, barrier_kernels,
                                        metal_ported=rel.as_posix() in metal_ported)
        dest.write_text(out)
        total_launches += n
        total_barrier += nb
        if renamed:
            all_renamed[rel.as_posix()] = renamed

    if not args.quiet:
        print(f'translate_cuda_to_cpu: {len(sources)} files, '
              f'{total_launches} kernel launches '
              f'({total_barrier} via the barrier dispatcher)')
        if barrier_kernels:
            print('  barrier dispatcher (uses __shared__ + __syncthreads): '
                  + ', '.join(sorted(barrier_kernels)))
        if atomics_only:
            print('  fast dispatcher despite __syncthreads (global atomics only): '
                  + ', '.join(sorted(atomics_only)))
        for rel, names in sorted(all_renamed.items()):
            print(f'  metal-ported, renamed to _cpu in {rel}: ' + ', '.join(names))
    return 0


if __name__ == '__main__':
    sys.exit(main())
