#!/usr/bin/env sh
# Download the prebuilt raylib 5.5 release for this platform into third_party/ (optional
# window renderer; see README.md "Install and build").
#
#   tools/get_raylib.sh              # -> third_party/raylib-5.5_macos or raylib-5.5_linux_amd64
#   tools/get_raylib.sh --print      # only print the URL and target directory
#   PR_RAYLIB=1 make build           # then build against it (or set RAYLIB_DIR=<dir>)
#
# Supported: macOS (universal), Linux x86_64. Elsewhere build raylib 5.5 from source and point
# RAYLIB_DIR at a directory holding include/raylib.h and lib/libraylib.a.
set -eu

VERSION=5.5
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "$(uname -s)" in
    Darwin) NAME="raylib-${VERSION}_macos" ;;
    Linux)
        case "$(uname -m)" in
            x86_64 | amd64) NAME="raylib-${VERSION}_linux_amd64" ;;
            *) echo "no prebuilt raylib ${VERSION} for Linux $(uname -m): build it from source and set RAYLIB_DIR" >&2; exit 1 ;;
        esac ;;
    *) echo "unsupported OS $(uname -s): build raylib ${VERSION} from source and set RAYLIB_DIR" >&2; exit 1 ;;
esac
URL="https://github.com/raysan5/raylib/releases/download/${VERSION}/${NAME}.tar.gz"
DEST="$ROOT/third_party"

if [ "${1:-}" = "--print" ]; then
    echo "$URL -> $DEST/$NAME"
    exit 0
fi
if [ -f "$DEST/$NAME/include/raylib.h" ] && [ -f "$DEST/$NAME/lib/libraylib.a" ]; then
    echo "raylib already present: $DEST/$NAME"
    exit 0
fi
mkdir -p "$DEST"
TMP="$DEST/.${NAME}.tar.gz"
echo "downloading $URL"
if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$TMP" "$URL"
else
    wget -O "$TMP" "$URL"
fi
tar -xzf "$TMP" -C "$DEST"
rm -f "$TMP"
test -f "$DEST/$NAME/include/raylib.h" && test -f "$DEST/$NAME/lib/libraylib.a"
echo "raylib ${VERSION} installed in $DEST/$NAME (build with: PR_RAYLIB=1 make build)"
