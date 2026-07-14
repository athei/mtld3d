#!/bin/sh
#
# Mechanical gate for the rules in docs/CONVENTIONS.md that a linter cannot
# express. clippy covers lint-shaped rules; these are the grep-shaped ones,
# and without a gate they drift. Run by `make audit`, which `make check` calls
# before every commit.
#
#   scripts/audit.sh                  audit the whole tree
#   scripts/audit.sh --file <path>    the per-file subset, for an editor hook
#   scripts/audit.sh --update-derives regenerate scripts/derive_inventory.txt
#
# POSIX sh + awk only: no bashisms, and no GNU awk extensions (`\s`, `\b`),
# since the stock macOS awk has neither.

set -eu

cd "$(git rev-parse --show-toplevel)"

INVENTORY=scripts/derive_inventory.txt

# Files permitted to hold each narrowly-scoped exception. These are sets, not
# counts: a new site fails even if an old one was deleted in the same change.
ALLOW_SITES='unix/unix/src/metal/command.rs
unix/unix/src/metal/macdrv.rs
windows/core/src/config.rs'

ONCELOCK_SITES='unix/unix/src/metal/blit.rs
unix/unix/src/metal/clear_quad.rs
unix/unix/src/metal/hdr_present.rs'

INLINE_ALWAYS_SITES='unix/shared/src/crumb.rs'

status=0

# Report a finding and name the section of docs/CONVENTIONS.md it comes from,
# so the fix is one lookup away rather than a guess.
report() {
    section=$1
    shift
    printf '\n== %s\n   docs/CONVENTIONS.md § %s\n' "$1" "$section" >&2
    shift
    printf '%s\n' "$@" | sed 's/^/   /' >&2
    status=1
}

# --- individual checks -------------------------------------------------------

# Every doc block of >= 2 lines is: title / empty doc line / body. Also caps
# every doc line at the 100 columns rustfmt gives code.
doc_shape() {
    awk '
        function text(l,   t) {
            t = l
            sub(/^[ \t]*/, "", t)
            sub(/^\/\/[\/!]/, "", t)
            return t
        }
        function is_doc(l,   t) {
            t = l
            sub(/^[ \t]*/, "", t)
            if (t ~ /^\/\/\/\//) return 0      # //// is a divider, not a doc
            if (t ~ /^\/\/\//)   return 1
            if (t ~ /^\/\/!/)    return 1
            return 0
        }
        function blank(l,   t) {
            t = text(l)
            gsub(/[ \t\r]/, "", t)
            return t == ""
        }
        FNR == 1 { run = 0 }
        {
            if (!is_doc($0)) { run = 0; next }
            run++
            if (length($0) > 100)
                printf "%s:%d: doc line is %d columns (max 100)\n", FILENAME, FNR, length($0)
            if (run == 1) {
                title_line = FNR
                title_blank = blank($0)
            }
            if (run == 2) {
                if (title_blank)
                    printf "%s:%d: doc block opens with an empty line; line 1 must be the title\n", \
                        FILENAME, title_line
                else if (!blank($0))
                    printf "%s:%d: line 2 of a doc block must be an empty doc line (title / blank / body)\n", \
                        FILENAME, FNR
            }
        }
    ' "$@"
}

# Every type deriving Clone and/or Copy, as `path Type Derives`. The committed
# inventory is diffed against this, so a speculative derive cannot slip in
# unnoticed: adding one means consciously recording it.
derive_scan() {
    awk '
        /^[ \t]*#\[derive\(/ {
            if ($0 ~ /Clone/ || $0 ~ /Copy/) {
                pending = $0
                pfile = FILENAME
            }
            next
        }
        pending != "" {
            # Attributes and doc comments may sit between the derive and the item.
            if ($0 ~ /^[ \t]*#\[/ || $0 ~ /^[ \t]*\/\//) next
            if (match($0, /(struct|enum|union)[ \t]+[A-Za-z_][A-Za-z0-9_]*/)) {
                name = substr($0, RSTART, RLENGTH)
                sub(/^(struct|enum|union)[ \t]+/, "", name)
                list = pending
                sub(/^[ \t]*#\[derive\(/, "", list)
                sub(/\)\].*$/, "", list)
                gsub(/[ \t]/, "", list)
                printf "%s %s %s\n", pfile, name, list
            }
            pending = ""
        }
    ' "$@" | LC_ALL=C sort
}

# A pattern that must not appear at all, anywhere. Mentioning it in a comment is
# fine — the rules are discussed in the prose of the very files they govern.
banned() {
    pattern=$1
    section=$2
    message=$3
    shift 3
    # -H, not just -n: with a single file grep omits the filename, and the
    # comment filter below keys on the `file:line:` prefix.
    hits=$(grep -HnE "$pattern" "$@" | grep -vE '^[^:]+:[0-9]+:[ \t]*//' || true)
    if [ -n "$hits" ]; then
        report "$section" "$message" "$hits"
    fi
}

# A pattern that must not appear in a COMMENT. The inverse of `banned`: these are
# release-hygiene rules, and the only place they can be broken is the prose.
banned_in_comments() {
    pattern=$1
    section=$2
    message=$3
    shift 3
    hits=$(grep -HnE "^[ \t]*(//|///|//!|#).*($pattern)" "$@" || true)
    if [ -n "$hits" ]; then
        report "$section" "$message" "$hits"
    fi
}

# A pattern confined to a known set of files: flag any file outside the set.
confined() {
    pattern=$1
    section=$2
    message=$3
    permitted=$4
    shift 4
    hits=$(grep -lE "$pattern" "$@" || true)
    strays=''
    for hit in $hits; do
        if ! printf '%s\n' "$permitted" | grep -qxF "$hit"; then
            strays="$strays$hit
"
        fi
    done
    if [ -n "$strays" ]; then
        report "$section" "$message" "$(printf '%s' "$strays")"
    fi
}

# --- modes -------------------------------------------------------------------

case "${1:-}" in
--update-derives)
    derive_scan $(git ls-files '*.rs') >"$INVENTORY"
    printf 'wrote %s (%d types)\n' "$INVENTORY" "$(wc -l <"$INVENTORY")"
    exit 0
    ;;
--file)
    file=${2:?--file needs a path}
    # An editor hook's view: only the checks that a single file can answer.
    case $file in *.rs) ;; *) exit 0 ;; esac
    [ -f "$file" ] || exit 0

    findings=$(doc_shape "$file")
    [ -z "$findings" ] || report 'Doc comments' 'doc-comment shape' "$findings"

    banned 'pub\(crate\)' 'No pub(crate) — use module hierarchy' \
        'pub(crate) visibility' "$file"
    banned 'extern "stdcall"' 'extern "system" everywhere, not extern "stdcall"' \
        'extern "stdcall"' "$file"
    banned 'msg_send!|class!\(|sel!\(' 'No raw msg_send! — use typed objc2-* bindings' \
        'untyped Obj-C selector' "$file"
    confined 'inline\(always\)' 'Inline attributes' \
        '#[inline(always)] outside the one measured site' "$INLINE_ALWAYS_SITES" "$file"
    confined '^[ \t]*static .*: *OnceLock' 'LazyLock over OnceLock' \
        'OnceLock static outside the runtime-argument sites' "$ONCELOCK_SITES" "$file"
    confined '#\[allow\(' 'Warning suppressions' \
        'lint suppression outside the three accepted per-site exceptions' "$ALLOW_SITES" "$file"

    exit $status
    ;;
-h | --help)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
esac

# --- whole-tree audit --------------------------------------------------------

set -- $(git ls-files '*.rs')

findings=$(doc_shape "$@")
if [ -n "$findings" ]; then
    report 'Doc comments' \
        "doc-comment shape: $(printf '%s\n' "$findings" | wc -l | tr -d ' ') findings" \
        "$findings"
fi

drift=$(derive_scan "$@" | diff "$INVENTORY" - || true)
if [ -n "$drift" ]; then
    report 'No default Copy / Clone on aggregate structs' \
        "derive inventory drift (< committed, > working tree). A Clone/Copy derive needs a concrete callsite; record it with scripts/audit.sh --update-derives" \
        "$drift"
fi

banned 'pub\(crate\)' 'No pub(crate) — use module hierarchy' \
    'pub(crate) visibility' "$@"
banned 'extern "stdcall"' 'extern "system" everywhere, not extern "stdcall"' \
    'extern "stdcall"' "$@"
banned 'msg_send!|class!\(|sel!\(' 'No raw msg_send! — use typed objc2-* bindings' \
    'untyped Obj-C selector' "$@"

confined 'inline\(always\)' 'Inline attributes' \
    '#[inline(always)] outside the one measured site' "$INLINE_ALWAYS_SITES" "$@"
confined '^[ \t]*static .*: *OnceLock' 'LazyLock over OnceLock' \
    'OnceLock static outside the runtime-argument sites' "$ONCELOCK_SITES" "$@"
confined '#\[allow\(' 'Warning suppressions' \
    'lint suppression outside the three accepted per-site exceptions' "$ALLOW_SITES" "$@"

modules=$(git ls-files '*/mod.rs' || true)
if [ -n "$modules" ]; then
    report 'Module style: foo.rs + foo/, not foo/mod.rs' 'mod.rs file' "$modules"
fi

# --- release hygiene ---------------------------------------------------------
#
# The repository is public. These rules were written down long before anything
# ran them, and every one of them had drifted by the time it was checked: the
# source described private work, cited the upstream test suite as if the reader
# could open it, and narrated its own development history.

RELEASE=$(git ls-files '*.rs' '*.msl' '*.md' '*.toml' '*.conf' Makefile |
    grep -v '^unix/conformance/' || true)

# shellcheck disable=SC2086
set -- $RELEASE

banned_in_comments 'dxvk|DXVK|dxmt|wined3d|wine3d|d9vk' \
    'Mechanical audit' \
    'reference to a competing D3D9 implementation — restate the contract in spec terms' "$@"

# Wine's d3d9 TEST suite, cited as provenance. A reader who clones this repo
# cannot open `device.c`, and the behavioural statement never needed it.
#
# Wine itself is fair game — this is the layer's host — so the net is drawn
# tight: the four d3d9 test-source stems, and a `test_*` name only when Wine is
# named on the same line. `server/thread.c` and a local `test_enable` binding
# both have to stay clean, and a wider pattern catches them.
banned_in_comments '(^|[^a-z_])(visual|device|stateblock|d3d9ex)\.c([^a-z]|$)|[Ww]ine.*\btest_[a-z_]{3,}|\btest_[a-z_]{3,}.*[Ww]ine' \
    'Mechanical audit' \
    'Wine test-file citation outside unix/conformance — keep the behaviour, drop the citation' "$@"

# Private development context: a reverse-engineered third-party DLL, and private
# shorthand for hardware that came up while debugging.
banned_in_comments 'reference perf DLL|reference.s x87|shim math kernel|\bFTV\b' \
    'Mechanical audit' \
    'reference to private (non-public) work — keep the technical claim, drop the provenance' "$@"

# The source is not a changelog. "commit" must be followed by an actual hash to
# count — otherwise "see committed state" reads as a reference to a commit.
banned_in_comments '\bbit us\b|[Uu]ntil commit|commit [0-9a-f]{7,}|20[0-9][0-9]-[01][0-9]-[0-3][0-9]' \
    'Mechanical audit' \
    'incident provenance — state the invariant, not the history that produced it' "$@"

banned_in_comments '[Cc]laude|[Aa]nthropic|Copilot|ChatGPT|generated with|(project|feedback)_[a-z0-9_]{4,}' \
    'Mechanical audit' \
    'tooling signal or internal note reference' "$@"

banned_in_comments '/Users/' \
    'Mechanical audit' \
    'absolute personal path' "$@"

if [ $status -eq 0 ]; then
    printf 'audit: clean (%d files)\n' "$(git ls-files '*.rs' | wc -l | tr -d ' ')"
fi
exit $status
