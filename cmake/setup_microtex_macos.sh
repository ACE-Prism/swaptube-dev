#!/bin/bash
# Clone and build MicroTeX on macOS, headless-only.
#
# Upstream's Unix build produces a single binary containing both the GTK editor
# and the -headless LaTeX->SVG converter, so it requires gtkmm plus
# gtksourceviewmm-3.0. gtksourceviewmm is deprecated and has no Homebrew
# formula, which makes that target unbuildable here. SwapTube only ever invokes
# the -headless path (see src/IO/Latex.cpp), and that path needs nothing but a
# Cairo SVG surface.
#
# This script therefore adds a headless-only target and migrates the Cairo
# backend from cairomm-1.0/pangomm-1.4 to the cairomm-1.16/pangomm-2.48 that
# Homebrew ships, where those enums became scoped. It is idempotent: re-running
# it re-clones from scratch.
set -e

DEST="${1:-../MicroTeX-master}"

for tool in cmake pkg-config git python3; do
    command -v "$tool" > /dev/null 2>&1 || { echo "setup_microtex_macos: missing required tool: $tool"; exit 1; }
done

for mod in cairomm-1.16 pangomm-2.48 tinyxml2; do
    pkg-config --exists "$mod" || {
        echo "setup_microtex_macos: missing dependency: $mod"
        echo "  brew install cairomm pangomm tinyxml2 fontconfig"
        exit 1
    }
done

echo ">>> Cloning MicroTeX into ${DEST}"
rm -rf "$DEST"
if ! git clone --depth 1 https://github.com/NanoMichael/MicroTeX.git "$DEST"; then
    # The GitHub git endpoint drops large clones on some networks; the codeload
    # tarball is a reliable fallback.
    echo ">>> git clone failed, falling back to the release tarball"
    rm -rf "$DEST"
    mkdir -p "$DEST"
    curl -fL --retry 5 --retry-delay 2 \
        https://github.com/NanoMichael/MicroTeX/archive/refs/heads/master.tar.gz \
        | tar xz -C "$DEST" --strip-components=1
fi

echo ">>> Applying the macOS headless patches"
python3 - "$DEST" <<'PY'
import pathlib, sys

root = pathlib.Path(sys.argv[1])

# 1. Let the Cairo backend compile outside the GTK build.
for rel in ('src/platform/cairo/graphic_cairo.cpp', 'src/platform/cairo/graphic_cairo.h'):
    p = root / rel
    s = p.read_text()
    old = '#if defined(BUILD_GTK) && !defined(MEM_CHECK)'
    new = '#if (defined(BUILD_GTK) || defined(BUILD_HEADLESS_ONLY)) && !defined(MEM_CHECK)'
    if old in s:
        p.write_text(s.replace(old, new, 1))

# 2. cairomm 1.0 / pangomm 1.4 kept these enums unscoped at namespace scope;
#    1.16 / 2.48 moved them into the owning class as `enum class`.
p = root / 'src/platform/cairo/graphic_cairo.cpp'
s = p.read_text()
renames = {
    'Cairo::FORMAT_ARGB32':   'Cairo::Surface::Format::ARGB32',
    'Cairo::LINE_CAP_BUTT':   'Cairo::Context::LineCap::BUTT',
    'Cairo::LINE_CAP_ROUND':  'Cairo::Context::LineCap::ROUND',
    'Cairo::LINE_CAP_SQUARE': 'Cairo::Context::LineCap::SQUARE',
    'Cairo::LINE_JOIN_BEVEL': 'Cairo::Context::LineJoin::BEVEL',
    'Cairo::LINE_JOIN_ROUND': 'Cairo::Context::LineJoin::ROUND',
    'Cairo::LINE_JOIN_MITER': 'Cairo::Context::LineJoin::MITER',
    'Cairo::LineCap':         'Cairo::Context::LineCap',
    'Cairo::LineJoin':        'Cairo::Context::LineJoin',
    'Pango::STYLE_NORMAL':    'Pango::Style::NORMAL',
    'Pango::STYLE_ITALIC':    'Pango::Style::ITALIC',
    'Pango::WEIGHT_NORMAL':   'Pango::Weight::NORMAL',
    'Pango::WEIGHT_BOLD':     'Pango::Weight::BOLD',
}
# Longest first, so Cairo::LineCap does not shadow Cairo::LINE_CAP_*.
for old in sorted(renames, key=len, reverse=True):
    s = s.replace(old, renames[old])
# FcFreeTypeQuery lives in a separate fontconfig header.
if 'fcfreetype.h' not in s:
    s = s.replace('#include <fontconfig/fontconfig.h>',
                  '#include <fontconfig/fontconfig.h>\n#include <fontconfig/fcfreetype.h>', 1)
p.write_text(s)

# 3. pkg-config imported targets carry the Homebrew library search path; the
#    bare library name only resolves on distros that install into /usr/lib.
p = root / 'CMakeLists.txt'
s = p.read_text()
s = s.replace('target_link_libraries(LaTeX PRIVATE tinyxml2)',
              'target_link_libraries(LaTeX PRIVATE PkgConfig::tinyxml2)')

# 4. Add the headless-only target ahead of the GTK branch.
gtk_branch = '''elseif (UNIX)
    message(STATUS "We are working with GTK on a Unix like OS")'''
headless_branch = '''elseif (HEADLESS_ONLY OR (APPLE AND NOT DEFINED HEADLESS_ONLY))
    message(STATUS "We are building the headless-only LaTeX converter")
    target_compile_definitions(LaTeX PUBLIC -DBUILD_HEADLESS_ONLY)
    find_package(Fontconfig REQUIRED)
    pkg_check_modules(CairoMM REQUIRED IMPORTED_TARGET cairomm-1.16)
    pkg_check_modules(PangoMM REQUIRED IMPORTED_TARGET pangomm-2.48)
    target_sources(LaTeX PRIVATE
            src/platform/cairo/graphic_cairo.cpp
            )
    target_link_libraries(LaTeX PUBLIC
            PkgConfig::CairoMM
            PkgConfig::PangoMM
            Fontconfig::Fontconfig
            )
    add_executable(LaTeXHeadless
            src/samples/headless_main.cpp
            )
    target_link_libraries(LaTeXHeadless PRIVATE LaTeX)
    set_target_properties(LaTeXHeadless PROPERTIES OUTPUT_NAME LaTeX)
''' + gtk_branch
if 'BUILD_HEADLESS_ONLY' not in s:
    s = s.replace(gtk_branch, headless_branch, 1)
p.write_text(s)
print("patches applied")
PY

echo ">>> Writing the headless entry point"
cat > "${DEST}/src/samples/headless_main.cpp" <<'EOF'
#include "config.h"

#if defined(BUILD_HEADLESS_ONLY) && !defined(MEM_CHECK)

// Headless-only entry point: LaTeX in, SVG out, no GUI toolkit.
// Mirrors the -headless mode of the stock gtkmm sample, minus the editor.

#include "latex.h"
#include "platform/cairo/graphic_cairo.h"
#include "atom/atom_basic.h"

#include <pangomm/init.h>
#include <cairomm/surface.h>
#include <cairomm/context.h>

#include <iostream>
#include <string>
#include <vector>

using namespace tex;

namespace {

struct Headless {
    std::string input;
    std::string outputFile;

    float textSize = 20.f;
    color foreground = BLACK;
    color background = TRANSPARENT;
    float padding = 10.f;
    float maxWidth = 720.f;

    int run() const {
        if (outputFile.empty()) {
            std::cerr << "Error: the option '-output' must be specified\n";
            return 1;
        }
        if (input.empty()) {
            std::cerr << "Error: the option '-input' must be specified\n";
            return 1;
        }

        auto* r = LaTeX::parse(utf82wide(input), maxWidth, textSize, textSize / 3.f, foreground);
        const float w = r->getWidth() + padding * 2;
        const float h = r->getHeight() + padding * 2;

        auto surface = Cairo::SvgSurface::create(outputFile, w, h);
        auto context = Cairo::Context::create(surface);
        Graphics2D_cairo g2(context);
        if (!isTransparent(background)) {
            g2.setColor(background);
            g2.fillRect(0, 0, w, h);
        }
        r->draw(g2, padding, padding);
        delete r;
        return 0;
    }
};

bool option(const std::string& arg, const char* name, std::string& out) {
    const std::string prefix = std::string(name) + "=";
    if (arg.rfind(prefix, 0) != 0) return false;
    out = arg.substr(prefix.size());
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    std::vector<std::string> opts;
    opts.reserve(argc);
    for (int i = 0; i < argc; i++) opts.emplace_back(argv[i]);

    Headless h;
    std::string value;
    for (const auto& arg : opts) {
        if (option(arg, "-input", h.input)) continue;
        if (option(arg, "-output", h.outputFile)) continue;
        if (option(arg, "-foreground", value)) { h.foreground = ColorAtom::getColor(value); continue; }
        if (option(arg, "-background", value)) { h.background = ColorAtom::getColor(value); continue; }
        if (option(arg, "-textsize", value)) { valueof(value, h.textSize); continue; }
        if (option(arg, "-padding", value)) { valueof(value, h.padding); continue; }
        if (option(arg, "-maxwidth", value)) { valueof(value, h.maxWidth); continue; }
    }

    if (h.textSize <= 0.f) h.textSize = 20.f;
    if (isTransparent(h.foreground)) h.foreground = BLACK;
    if (h.maxWidth <= 0.f) h.maxWidth = 720.f;

    Pango::init();
    LaTeX::init();
    const int result = h.run();
    LaTeX::release();
    return result;
}

#endif
EOF

echo ">>> Building"
mkdir -p "${DEST}/build"
cd "${DEST}/build"
cmake .. > /dev/null
make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

if [ ! -s LaTeX ]; then
    echo "setup_microtex_macos: build finished but ${DEST}/build/LaTeX is missing"
    exit 1
fi

echo ">>> Verifying"
./LaTeX -headless -foreground=#ff000000 "-input=x^2" -output=/tmp/microtex_selftest.svg > /dev/null 2>&1
if [ ! -s /tmp/microtex_selftest.svg ]; then
    echo "setup_microtex_macos: the binary built but produced no SVG"
    exit 1
fi
rm -f /tmp/microtex_selftest.svg
echo ">>> MicroTeX is ready at ${DEST}/build/LaTeX"
