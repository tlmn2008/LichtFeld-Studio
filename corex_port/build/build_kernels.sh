#!/bin/bash
# CoreX ivcore11 build of LichtFeld-Studio LibTorch-free CUDA kernels.
# nvcc -> clang++ -x ivcore ; arch -> ivcore11 ; nvcc-only flags stripped.
set -u
cd "$(dirname "$0")/../.."   # repo root
. corex_port/corex_env.sh

REPO=$(pwd)
OBJDIR=corex_port/build/obj
mkdir -p "$OBJDIR"
LOG=corex_port/build/compile.log
: > "$LOG"

INC="-I $REPO/src/core/include -I $REPO/src/core/tensor -I $REPO/src/diagnostics/include -I $REPO/src -I $REPO/external -I /usr/local/corex/include"
# Release flags translated from CMake Linux branch (-O3 -use_fast_math --extended-lambda
# --expt-relaxed-constexpr): clang supports extended lambda / relaxed constexpr natively;
# -use_fast_math -> -ffast-math ; nvcc-only long-opts dropped.
CUFLAGS="-x ivcore --cuda-path=/usr/local/corex --cuda-gpu-arch=ivcore11 -std=c++20 -O3 -ffast-math -DLFS_CORE_EXPORTS -DNDEBUG"

# List of source files passed as args; each compiled to an object.
ok=0; fail=0; failed_list=""
for src in "$@"; do
    base=$(echo "$src" | tr '/' '_')
    obj="$OBJDIR/${base}.o"
    echo "======== COMPILE $src ========" | tee -a "$LOG"
    if clang++ $CUFLAGS -c "$src" -o "$obj" $INC >>"$LOG" 2>&1; then
        echo "RESULT: OK $src" | tee -a "$LOG"
        ok=$((ok+1))
    else
        echo "RESULT: FAIL $src" | tee -a "$LOG"
        fail=$((fail+1)); failed_list="$failed_list $src"
    fi
done
echo "" | tee -a "$LOG"
echo "SUMMARY: ok=$ok fail=$fail" | tee -a "$LOG"
[ -n "$failed_list" ] && echo "FAILED:$failed_list" | tee -a "$LOG"
exit $fail
