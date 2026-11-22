#!/bin/sh
# Capture or replay the baseline invocation matrix in test/baseline_cases.txt.
#
#   baseline.sh capture DIALECT OUTDIR MINDGRAB_EXE M16_EXE
#   baseline.sh compare DIALECT OUTDIR MINDGRAB_EXE M16_EXE
#
# capture runs every case and writes OUTDIR/<case>/ plus OUTDIR/manifest.sha256.
# compare re-runs every case against the given executables and checks the result
# against that manifest: bit-exact where the case's tolerance is 0, and a
# voxel-count comparison against the retained output where it is not.
#
# DIALECT is legacy (the multi-model brainchopc) or modular (one executable per
# model). It selects which flag column of the case file is used, which is what
# lets a single matrix survive the CLI change in milestone 4.
#
# The point of this script is that every milestone is checked against a LIVE
# previous binary, not only against files captured once at the start.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cases="$here/baseline_cases.txt"

if command -v sha256sum >/dev/null 2>&1; then
	sha256() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
	sha256() { shasum -a 256 "$@"; }
else
	echo "baseline.sh: neither sha256sum nor shasum is available" >&2
	exit 2
fi

usage() {
	echo "usage: baseline.sh capture|compare DIALECT OUTDIR MINDGRAB_EXE M16_EXE" >&2
	exit 2
}

[ $# -eq 5 ] || usage
mode=$1
dialect=$2
outdir=$3
exe_mindgrab=$4
exe_m16=$5

case $mode in capture|compare) ;; *) usage ;; esac
case $dialect in legacy|modular) ;; *) usage ;; esac
[ -f "$here/t1_crop.nii.gz" ] || {
	echo "baseline.sh: missing $here/t1_crop.nii.gz" >&2; exit 2; }

# The @extra/ cases run against real acquisitions that are deliberately not
# tracked. Absent BC_EXTRA_DIR they skip, so the same case file works on a fresh
# clone and on a machine that has them.
extra_dir=${BC_EXTRA_DIR:-}
[ -n "$extra_dir" ] || echo "baseline.sh: BC_EXTRA_DIR unset; @extra cases will skip"

manifest="$outdir/manifest.sha256"
# compare must not overwrite the captured outputs it is comparing against, so it
# replays into a scratch directory alongside them.
if [ "$mode" = compare ]; then
	work="$outdir.replay-$$"
	trap 'rm -rf "$work"' EXIT INT TERM
else
	work="$outdir"
fi
mkdir -p "$work"

# Digest every file a case produced, path-relative and in a stable order, so a
# case that writes extra outputs needs no special handling here.
digest_case() {
	dir=$1
	( cd "$dir" && find . -type f | LC_ALL=C sort | while IFS= read -r file; do
		sha256 "$file"
	done )
}

pass=0; fail=0; skip=0
failed_cases=""
: >"$work/manifest.new"

while IFS='|' read -r name model input legacy modular retain tol; do
	case $name in ''|\#*) continue ;; esac

	if [ "$dialect" = legacy ]; then flags=$legacy; else flags=$modular; fi
	if [ "$flags" = "-" ]; then
		printf '  SKIP %-26s (no %s equivalent)\n' "$name" "$dialect"
		skip=$((skip + 1))
		continue
	fi

	if [ "$model" = mindgrab ]; then exe=$exe_mindgrab; else exe=$exe_m16; fi
	dir="$work/$name"
	rm -rf "$dir"; mkdir -p "$dir"
	# @DIR@ lets a case name a secondary output (--mask) inside its own
	# directory, which is what keeps digest_case free of per-case knowledge.
	flags=$(printf '%s' "$flags" | sed "s#@DIR@#$dir#g")

	case $input in
	@extra/*)
		if [ -z "$extra_dir" ]; then
			printf '  SKIP %-26s (BC_EXTRA_DIR unset)\n' "$name"
			skip=$((skip + 1)); rm -rf "$dir"; continue
		fi
		image="$extra_dir/${input#@extra/}.nii.gz"
		if [ ! -f "$image" ]; then
			printf '  SKIP %-26s (no %s)\n' "$name" "$image"
			skip=$((skip + 1)); rm -rf "$dir"; continue
		fi
		;;
	*)
		image="$here/$input.nii.gz"
		if [ ! -f "$image" ]; then
			printf '  FAIL %-26s (missing input %s)\n' "$name" "$image"
			fail=$((fail + 1)); failed_cases="$failed_cases $name"
			continue
		fi
		;;
	esac

	# shellcheck disable=SC2086 -- flags is a deliberately word-split argument list
	if ! "$exe" $flags -o "$dir/out.nii.gz" "$image" >"$dir/.stdout" 2>"$dir/.stderr"; then
		if grep -q 'unavailable' "$dir/.stderr"; then
			printf '  SKIP %-26s (backend unavailable)\n' "$name"
			skip=$((skip + 1))
			rm -rf "$dir"
			continue
		fi
		printf '  FAIL %-26s (exit status)\n' "$name"
		sed 's/^/       /' "$dir/.stderr" >&2
		fail=$((fail + 1)); failed_cases="$failed_cases $name"
		continue
	fi
	rm -f "$dir/.stdout" "$dir/.stderr"

	digest_case "$dir" | sed "s#^#$name #" >>"$work/manifest.new"

	if [ "$mode" = capture ]; then
		[ "$retain" = full ] || rm -f "$dir"/*.nii.gz
		printf '  ok   %-26s\n' "$name"
		pass=$((pass + 1))
		continue
	fi

	# compare: bit-exact via digests first, then fall back to the tolerance.
	expected=$(grep "^$name " "$manifest" || true)
	actual=$(grep "^$name " "$work/manifest.new" || true)
	if [ -z "$expected" ]; then
		printf '  FAIL %-26s (absent from baseline manifest)\n' "$name"
		fail=$((fail + 1)); failed_cases="$failed_cases $name"
	elif [ "$expected" = "$actual" ]; then
		printf '  ok   %-26s bit-exact\n' "$name"
		pass=$((pass + 1))
	elif [ "$tol" -gt 0 ] && [ -f "$outdir/$name/out.nii.gz" ]; then
		if python3 "$here/nii_diff.py" "$outdir/$name/out.nii.gz" \
				"$dir/out.nii.gz" "$tol" >"$dir/.diff" 2>&1; then
			printf '  ok   %-26s %s\n' "$name" "$(cat "$dir/.diff")"
			pass=$((pass + 1))
		else
			printf '  FAIL %-26s %s\n' "$name" "$(cat "$dir/.diff")"
			fail=$((fail + 1)); failed_cases="$failed_cases $name"
		fi
	else
		printf '  FAIL %-26s output changed\n' "$name"
		fail=$((fail + 1)); failed_cases="$failed_cases $name"
	fi
	[ "$retain" = full ] || rm -f "$dir"/*.nii.gz
done <"$cases"

if [ "$mode" = capture ]; then
	mv "$work/manifest.new" "$manifest"
	echo "baseline: captured $pass cases, skipped $skip, failed $fail -> $manifest"
else
	echo "baseline: $pass passed, $skip skipped, $fail failed"
fi
[ "$fail" -eq 0 ] || { echo "baseline: failing cases:$failed_cases" >&2; exit 1; }
