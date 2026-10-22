#!/bin/sh
# Clone the pinned vLLM (source.env) into vllm-sim/vllm and apply patches/.
# An existing checkout is left untouched.
#
# After committing changes in the checkout, record them as patches with:
#   . vllm-sim/source.env && rm -f vllm-sim/patches/*.patch &&
#   git -C vllm-sim/vllm format-patch --no-signature --zero-commit --no-numbered -o ../patches "$VLLM_REVISION..HEAD"
set -eu
here=$(cd "$(dirname "$0")" && pwd)
. "$here/source.env"
checkout=$here/vllm
if [ -e "$checkout" ]; then
    echo "$checkout already exists; leaving it as is"
    exit 0
fi
git clone --depth 1 --branch "$VLLM_BRANCH" "$VLLM_REPOSITORY" "$checkout"
git -C "$checkout" fetch --depth 1 origin "$VLLM_REVISION"
git -C "$checkout" checkout -q -B bench "$VLLM_REVISION"
git -C "$checkout" am -q "$here"/patches/*.patch
echo "vLLM $VLLM_REVISION with:" && git -C "$checkout" log --oneline "$VLLM_REVISION..HEAD"
