#!/usr/bin/env python3
"""Generate the Metal backend's host/device plumbing from the CUDA kernel signatures.

src/CUDA stays the single source of truth. A CUDA launch passes its arguments by
value automatically; Metal has no equivalent, and its 31-buffer limit rules out
one binding per argument anyway (fractal_2d.cu's `go` kernel takes 148 of them).
So every kernel gets one generated struct holding all of its by-value arguments,
bound as a single `constant` buffer at index 0, with the pointer arguments bound
at 1..n.

Generating that struct rather than writing it by hand is what keeps the host and
the shader from disagreeing about the layout: both include the same header, and
upstream adding a kernel parameter turns into a compile error in the host wrapper
instead of silent garbage on the GPU.

Two outputs, both into src/Metal/generated/:

  kernel_args.h     the argument structs and buffer index constants
  metal_sources.h   the .metal files embedded as string literals

The sources are embedded because the Metal offline compiler (xcrun metal) ships
only with full Xcode. Compiling from source at runtime through
MTLDevice::newLibrary works with nothing but the Command Line Tools installed,
and embedding avoids depending on a shader file's path at runtime.
"""

import argparse
import hashlib
import pathlib
import re
import subprocess
import sys

KERNEL_DEF = re.compile(r'__global__\s+void\s+(?P<name>\w+)\s*\(')

# Types that are plain scalars on both sides and need no namespace handling.
SCALARS = {
    'bool', 'char', 'signed char', 'unsigned char', 'short', 'unsigned short',
    'int', 'unsigned', 'unsigned int', 'long', 'unsigned long', 'long long',
    'unsigned long long', 'float', 'size_t',
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
}


def strip_comments(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', text)


def matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('unbalanced parentheses in kernel signature')


def matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def split_params(params: str):
    """Split a parameter list on top-level commas."""
    out, depth, current = [], 0, []
    for ch in params:
        if ch in '(<[':
            depth += 1
        elif ch in ')>]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        out.append(''.join(current))
    return [p.strip() for p in out if p.strip()]


class Param:
    def __init__(self, decl: str):
        self.decl = decl
        self.is_pointer = '*' in decl

        text = decl.replace('const', ' ').replace('&', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        # The name is the trailing identifier; everything before it is the type.
        m = re.match(r'^(?P<type>.*?)(?P<name>\w+)\s*(?P<array>\[\s*\w*\s*\])?$', text)
        if not m:
            raise ValueError(f'cannot parse parameter: {decl!r}')
        self.name = m.group('name')
        self.array = m.group('array') or ''
        self.type = m.group('type').strip()

        # The struct is emitted inside SHARED_FILE_PREFIX, which is the Cuda
        # namespace on the device and nothing on the host, so the qualifier has
        # to come off for the same spelling to work on both sides.
        self.type = self.type.replace('Cuda::', '').strip()

    def field(self) -> str:
        return f'{self.type} {self.name}{self.array};'

    def pointee(self) -> str:
        return self.type.replace('*', '').strip()


class Kernel:
    def __init__(self, name: str, params, source: pathlib.Path, body_hash: str):
        self.name = name
        self.params = params
        self.source = source
        self.body_hash = body_hash

    @property
    def by_value(self):
        return [p for p in self.params if not p.is_pointer]

    @property
    def pointers(self):
        return [p for p in self.params if p.is_pointer]

    @property
    def struct_name(self):
        # runRaymarch -> RunRaymarchArgs, argb_to_p010 -> ArgbToP010Args
        parts = re.split(r'[_\s]+', re.sub(r'(?<!^)(?=[A-Z])', '_', self.name))
        return ''.join(p[:1].upper() + p[1:] for p in parts if p) + 'Args'


def parse_kernels(path: pathlib.Path):
    raw = path.read_text()
    text = strip_comments(raw)

    # The hash covers the file's __device__ helpers as well as the kernel body,
    # because a port depends on both: changing distMap() changes what the shader
    # should compute just as surely as changing the kernel. Comments and
    # whitespace are normalized away so reformatting is not reported as drift.
    first_global = text.find('__global__')
    prefix = text if first_global < 0 else text[:first_global]
    prefix_normalized = re.sub(r'\s+', ' ', prefix)

    kernels = []
    for m in KERNEL_DEF.finditer(text):
        open_paren = m.end() - 1
        close_paren = matching_paren(text, open_paren)
        params = [Param(p) for p in split_params(text[open_paren + 1:close_paren])]

        brace = text.find('{', close_paren)
        body = text[brace:matching_brace(text, brace)] if brace >= 0 else ''
        digest = hashlib.sha256((prefix_normalized + re.sub(r'\s+', ' ', body)).encode())

        kernels.append(Kernel(m.group('name'), params, path, digest.hexdigest()[:16]))
    return kernels


def emit_device_prefix(path: pathlib.Path, out_path: pathlib.Path):
    """Extract a .cu file's device-side prefix for the shader to include.

    Everything before the first __global__ is the constants and __device__ helpers,
    which Metal can compile as-is once the prelude has defined the qualifiers.
    Everything from __global__ onward is the kernel entry and the host wrapper: the
    entry has to be rewritten for Metal's argument model, and the wrapper uses
    std::complex and <<<>>>, neither of which belongs in a shader.

    Extracting rather than hand-copying keeps these helpers following src/CUDA, so
    a port only owns the few lines of its kernel entry point.
    """
    src = path.read_text()
    cut = src.find('__global__')
    prefix = src if cut < 0 else src[:cut]

    # These two define the pixel accessors against a plain uint32_t* and reach for
    # atomicCAS, neither of which exists in Metal. MetalPrelude.h provides
    # px_load/px_store/px_cas over device atomic_uint instead, so the includes are
    # dropped rather than translated. A kernel that needs the circle and line
    # drawing in common_graphics.cuh will need MSL versions of those written.
    for replaced in ('color.cuh', 'common_graphics.cuh'):
        prefix = re.sub(rf'^\s*#include\s+"[^"]*{re.escape(replaced)}"\s*$',
                        f'// #include "{replaced}" -- replaced by MetalPrelude.h',
                        prefix, flags=re.M)

    header = (
        f'// GENERATED by cmake/generate_metal_bindings.py from src/CUDA/{path.name}\n'
        f'// -- do not edit. The device-side prefix of that file: everything ahead of\n'
        f'// its first __global__.\n'
        f'#pragma once\n\n'
    )
    out_path.write_text(header + prefix)


def emit_args_header(kernels, out_path: pathlib.Path):
    lines = [
        '// GENERATED by cmake/generate_metal_bindings.py -- do not edit.',
        '//',
        '// One struct per ported kernel, holding the arguments a CUDA launch would',
        '// have passed by value. Included by both src/Metal/*.metal and the host',
        '// wrappers, so the two cannot disagree about the layout.',
        '//',
        '// SHARED_FILE_PREFIX puts these in the Cuda namespace for the shader and at',
        '// global scope for the host, matching how the rest of Host_Device_Shared',
        '// behaves. The field types are spelled unqualified so one definition serves',
        '// both.',
        '#pragma once',
        '',
        '#include "../../Host_Device_Shared/vec.h"',
        '#include "../../Host_Device_Shared/shared_precompiler_directives.h"',
    ]

    # Kept self-contained for both sides: several host wrappers include this, and
    # any of them naming a cuComplex field would otherwise have to remember to
    # define it first. The shader already has it from MetalPrelude.h.
    if any(p.type.startswith('cu') for k in kernels for p in k.by_value):
        lines += [
            '',
            '#ifndef __METAL_VERSION__',
            '#include "../../CPUBackend/cuComplex.h"',
            '#endif',
        ]

    lines += [
        '',
        'SHARED_FILE_PREFIX',
        '',
    ]

    for k in kernels:
        lines.append(f'// {k.name}  ({k.source.as_posix()})')
        lines.append(f'struct {k.struct_name} {{')
        for p in k.by_value:
            lines.append(f'    {p.field()}')
        if not k.by_value:
            lines.append('    int unused_placeholder;')
        lines.append('};')
        if k.pointers:
            lines.append('// Buffer bindings: index 0 is the args struct above.')
            for i, p in enumerate(k.pointers, start=1):
                lines.append(f'//   [[buffer({i})]] {p.type} {p.name}')

        # Unpacking into locals lets a ported kernel keep the CUDA body verbatim,
        # bare parameter names and all, and moves the values out of the constant
        # address space that vec.h's thread references cannot bind to.
        if k.by_value:
            lines.append(f'#ifdef __METAL_VERSION__')
            lines.append(f'#define SWAPTUBE_UNPACK_{k.struct_name}(a) \\')
            for n, p in enumerate(k.by_value):
                cont = ' \\' if n + 1 < len(k.by_value) else ''
                lines.append(f'    const {p.type} {p.name} = (a).{p.name};{cont}')
            lines.append('#endif')
        lines.append('')

    lines += ['SHARED_FILE_SUFFIX', '']
    out_path.write_text('\n'.join(lines))


def flatten_includes(source: pathlib.Path, include_root: pathlib.Path, stubs: pathlib.Path) -> str:
    """Run the C preprocessor over a .metal file to inline its quoted includes.

    Runtime compilation hands MTLDevice::newLibrary a bare source string with no
    filesystem context, so `#include "../../Host_Device_Shared/vec.h"` would not
    resolve. clang -E does that resolution at build time. It only needs to lex
    the file, so MSL-specific syntax like [[buffer(0)]] passes through untouched.

    __METAL_VERSION__ is defined here so the shared headers take their Metal
    branch and skip the C++ standard library includes, which is also why
    <metal_stdlib> must not appear in these files: the runtime prepends it.

    The stub directory satisfies the <cuda_runtime.h> and <thrust/complex.h>
    includes that the reused .cuh headers carry. -nostdsysteminc keeps a stray
    <stdio.h> from dragging the entire C library into the shader source.
    """
    result = subprocess.run(
        ['clang', '-E', '-P', '-x', 'c++', '-std=c++17',
         '-D__METAL_VERSION__=310',
         '-Xclang', '-nostdsysteminc',
         '-I', str(stubs),
         '-I', str(include_root),
         # The extracted device prefixes keep the .cu's own relative includes,
         # which resolved from src/CUDA rather than from src/Metal/generated.
         '-I', str(include_root / 'CUDA'),
         str(source)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'preprocessing {source} failed:\n{result.stderr}')
    return result.stdout


# A parameter that is a reference: `const Cuda::vec3& pos`, `float& zr`.
#
# The trailing name must start with a letter and the ampersand must have
# whitespace on one side, which together keep expressions from matching: neither
# `col&0xff000000` nor `a&b` qualifies, while every reference parameter in this
# codebase does.
REFERENCE_PARAM = re.compile(
    r'^\s*(?P<decl>(?:const\s+)?[A-Za-z_][\w:]*(?:<[^<>]*>)?)'
    r'(?P<amp>\s+&\s*|\s*&\s+)(?P<name>[A-Za-z_]\w*)\s*$')

# A pointer parameter. Every one in this codebase's device code addresses a buffer
# the kernel was handed, so `device` is the right space; the prelude qualifies its
# own `thread` pointers explicitly.
#
# `T* name` and `a * b` are the same shape, and C++ only tells them apart by
# knowing whether T is a type, so the leading token is checked against the types
# that actually appear. Without that, `sqrt(x*y)` reads as a declaration.
POINTER_PARAM = re.compile(
    r'^\s*(?P<decl>(?:const\s+)?(?P<type>[A-Za-z_][\w:]*(?:<[^<>]*>)?))'
    r'(?P<star>\s*\*\s*)(?P<name>[A-Za-z_]\w*)\s*$')

POINTER_TYPES = SCALARS | {
    'void', 'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4', 'quat',
    'mat2', 'mat3', 'mat4', 'Bitboard', 'cuComplex', 'cuFloatComplex',
}


def is_pointer_type(name: str) -> bool:
    bare = name.split('::')[-1].split('<')[0]
    return bare in POINTER_TYPES or bare.endswith('_t') or '::' in name

# Something that can introduce a parameter list: a plain function or constructor
# name, or an operator overload.
CALLABLE = re.compile(
    r'(?:\boperator\s*(?:\[\s*\]|\(\s*\)|[-+*/%^&|~!=<>]+)\s*|\b[A-Za-z_]\w*\s*)\(')

# A function whose return type is a reference, such as the compound assignment
# operators in vec.h: `inline vec2& operator+=(...)`. Metal needs an address
# space on the return type too.
REFERENCE_RETURN = re.compile(
    r'^(?P<lead>\s*(?:(?:inline|static|constexpr|HOST_DEVICE|__device__|__forceinline__)\s+)*)'
    r'(?P<type>[A-Za-z_][\w:]*)\s*&\s*'
    r'(?P<name>operator\s*(?:\[\s*\]|\(\s*\)|[-+*/%^&|~!=<>]+)|[A-Za-z_]\w*)\s*\(')


def qualify_reference_params(source: str) -> str:
    """Insert Metal's `thread` address space on reference parameters.

    Metal requires an address space on every pointer and reference parameter and
    offers no way to be generic over it. Every reference in this codebase's device
    code is a small value-like struct or an out-parameter backed by a local, so
    `thread` is always the right answer, and the CUDA sources can be reused
    verbatim instead of being hand-ported.

    Only parameter lists of definitions and declarations are touched -- the
    closing parenthesis has to be followed by `{` or `;` -- so call sites and the
    bitwise arithmetic inside them are left alone. Anything this misses is a hard
    error from the Metal compiler rather than silent breakage.
    """
    out = []
    i = 0
    for m in CALLABLE.finditer(source):
        if m.start() < i:
            continue  # inside a span already rewritten
        try:
            close = matching_paren(source, m.end() - 1)
        except ValueError:
            continue

        # What follows the closing parenthesis tells a declaration from a call:
        # a body, a prototype's semicolon, or a constructor's initializer list.
        after = source[close + 1:close + 40].lstrip()
        for qualifier in ('const', 'noexcept'):
            if after.startswith(qualifier):
                after = after[len(qualifier):].lstrip()
        if not after[:1] in ('{', ';', ':'):
            continue

        params = source[m.end():close]
        rewritten = []
        for part in split_params(params):
            ref = REFERENCE_PARAM.match(part)
            if ref:
                rewritten.append(f'thread {ref.group("decl")}& {ref.group("name")}')
                continue
            ptr = POINTER_PARAM.match(part)
            if (ptr and is_pointer_type(ptr.group('type'))
                    and not any(s in part for s in ('thread', 'device', 'constant'))):
                rewritten.append(f'device {ptr.group("decl")}* {ptr.group("name")}')
                continue
            rewritten.append(part)

        if rewritten and params.strip():
            out.append(source[i:m.end()])
            out.append(', '.join(p.strip() for p in rewritten))
            i = close

    out.append(source[i:])
    return promote_file_scope_constants(qualify_reference_returns(''.join(out)))


# A file-scope constant such as `const float bailout_radius = 256;`, which the
# .cu files use freely and Metal requires in the constant address space.
FILE_SCOPE_CONSTANT = re.compile(
    r'^(?P<lead>\s*)(?:static\s+)?const\s+(?P<type>[A-Za-z_][\w:]*)\s+'
    r'(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;{}]+);\s*$')


def promote_file_scope_constants(source: str) -> str:
    """Move file-scope constants into Metal's constant address space.

    Tracked by brace depth so only genuine file-scope declarations are touched;
    the same syntax inside a function body is an ordinary local and must stay one.
    """
    lines = source.split('\n')
    depth = 0
    for n, line in enumerate(lines):
        if depth == 0:
            m = FILE_SCOPE_CONSTANT.match(line)
            if m:
                lines[n] = (f'{m.group("lead")}constant {m.group("type")} '
                            f'{m.group("name")} = {m.group("value").strip()};')
        depth += line.count('{') - line.count('}')
    return '\n'.join(lines)


def qualify_reference_returns(source: str) -> str:
    """Insert `thread` on functions that return a reference.

    Line-wise is enough: every such declaration in this codebase -- the compound
    assignment operators in vec.h -- fits on one line, and a return type only
    ever appears at the start of a declaration.
    """
    lines = source.split('\n')
    for n, line in enumerate(lines):
        m = REFERENCE_RETURN.match(line)
        if m:
            lines[n] = f'{m.group("lead")}thread {m.group("type")}& {m.group("name")}(' + line[m.end():]
    return '\n'.join(lines)


def emit_sources_header(metal_dir: pathlib.Path, include_root: pathlib.Path,
                        stubs: pathlib.Path, out_path: pathlib.Path):
    """Embed each flattened .metal source as its own string literal.

    One entry per shader file, compiled into its own MTLLibrary at startup. They
    cannot be concatenated into a single library: each file is flattened
    independently, so the shared headers appear in every one of them and the
    duplicate struct definitions would collide. Separate libraries also mirror how
    nvcc compiles each .cu on its own.
    """
    sources = sorted(metal_dir.glob('*.metal'))
    lines = [
        '// GENERATED by cmake/generate_metal_bindings.py -- do not edit.',
        '//',
        '// The shader sources with their includes already flattened, embedded so the',
        '// runtime can compile them through MTLDevice::newLibrary. That path needs no',
        '// offline compiler, which matters because xcrun metal ships only with full',
        '// Xcode, and it needs no shader file to be findable at runtime.',
        '#pragma once',
        '',
        'namespace swaptube_metal {',
        '',
        'struct EmbeddedShader {',
        '    const char* name;',
        '    const char* source;',
        '};',
        '',
        'inline const EmbeddedShader* embedded_shaders(int& count) {',
        '    static const EmbeddedShader shaders[] = {',
    ]

    for src in sources:
        flattened = qualify_reference_params(flatten_includes(src, include_root, stubs))
        lines.append(f'        {{"{src.name}",')
        lines.append(f'         R"SWAPTUBE_MSL({flattened})SWAPTUBE_MSL"}},')

    if not sources:
        lines.append('        {nullptr, nullptr},')

    lines += [
        '    };',
        f'    count = {len(sources)};',
        '    return shaders;',
        '}',
        '',
        '} // namespace swaptube_metal',
        '',
    ]
    out_path.write_text('\n'.join(lines))


def read_manifest(path: pathlib.Path):
    """file -> {kernel: body_hash} recorded when the kernel was ported."""
    ported, hashes = [], {}
    if not path.exists():
        return ported, hashes
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        ported.append(parts[0])
        for entry in parts[1:]:
            kernel, _, h = entry.partition('=')
            hashes[kernel] = h
    return ported, hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', required=True, type=pathlib.Path)
    ap.add_argument('--metal', required=True, type=pathlib.Path)
    ap.add_argument('--manifest', required=True, type=pathlib.Path)
    ap.add_argument('--stubs', required=True, type=pathlib.Path,
                    help='directory of empty CUDA headers for the include flattener')
    ap.add_argument('--check', action='store_true',
                    help='report kernels whose CUDA body changed since they were ported')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    ported, recorded = read_manifest(args.manifest)
    if not ported:
        print('generate_metal_bindings: manifest lists no ported files', file=sys.stderr)

    kernels = []
    missing = []
    for rel in ported:
        path = args.cuda / rel
        if not path.exists():
            missing.append(rel)
            continue
        kernels.extend(parse_kernels(path))

    if missing:
        print(f'generate_metal_bindings: manifest references missing files: {missing}', file=sys.stderr)
        return 1

    # Drift: the argument structs regenerate from src/CUDA every build, so a
    # changed signature already breaks the host wrapper's compile. A changed
    # *body* is invisible to the compiler, so it is hashed and compared here.
    drifted = [k for k in kernels if k.name in recorded and recorded[k.name] != k.body_hash]
    unrecorded = [k for k in kernels if k.name not in recorded]

    if args.check:
        for k in drifted:
            print(f'generate_metal_bindings: DRIFT {k.source.as_posix()}::{k.name} '
                  f'changed since it was ported to Metal '
                  f'(recorded {recorded[k.name]}, now {k.body_hash})', file=sys.stderr)
        for k in unrecorded:
            print(f'generate_metal_bindings: {k.source.as_posix()}::{k.name} has no '
                  f'recorded hash; add {k.name}={k.body_hash} to the manifest', file=sys.stderr)
        return 1 if drifted else 0

    generated = args.metal / 'generated'
    generated.mkdir(parents=True, exist_ok=True)
    emit_args_header(kernels, generated / 'kernel_args.h')
    for rel in ported:
        path = args.cuda / rel
        emit_device_prefix(path, generated / f'{path.stem}_device.h')
    emit_sources_header(args.metal, args.cuda.parent, args.stubs, generated / 'metal_sources.h')

    if not args.quiet:
        shaders = len(list(args.metal.glob('*.metal')))
        print(f'generate_metal_bindings: {len(ported)} ported file(s), {len(kernels)} kernel(s), '
              f'{shaders} shader source(s) embedded')
        for k in drifted:
            print(f'  WARNING: {k.source.as_posix()}::{k.name} drifted from its Metal port '
                  f'(recorded {recorded[k.name]}, now {k.body_hash})')
        for k in unrecorded:
            print(f'  note: {k.name} has no recorded hash (add {k.name}={k.body_hash})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
