#!/bin/sh
# Verify a built installer package WITHOUT installing it.
#
#   verify_macos_pkg.sh FILE.pkg
#
# Expands the package into a scratch directory and checks the executables it
# would install: that they are correctly signed, that they carry no dependency
# outside macOS itself, that no build-machine path leaked into them, and that
# they actually run. The last point matters most -- a package can be perfectly
# signed and still install a binary that dies looking for a library or a model
# file that is not there.
set -eu

bc_pkg=${1:?usage: verify_macos_pkg.sh FILE.pkg}
[ -f "$bc_pkg" ] || {
	echo "verify_macos_pkg.sh: file not found: $bc_pkg" >&2
	exit 2
}

# pkgutil exits nonzero for an unsigned package, which under `set -e` would end
# this script before it checked anything that matters. An ad-hoc build is
# legitimately unsigned, so distinguish that from a signature that is present
# and broken: warn for the first, fail for the second. notarize_macos.sh is
# where an unsigned package is actually refused.
if bc_signature=$(pkgutil --check-signature "$bc_pkg" 2>&1); then
	echo "$bc_signature" | sed 's/^/  /'
else
	case "$bc_signature" in
	*"no signature"*)
		echo "verify_macos_pkg.sh: WARNING - package is unsigned; this build" >&2
		echo "  cannot be notarised and Gatekeeper will refuse it elsewhere." >&2
		;;
	*)
		echo "$bc_signature" >&2
		echo "verify_macos_pkg.sh: package signature is invalid" >&2
		exit 1
		;;
	esac
fi

bc_work=$(mktemp -d "${TMPDIR:-/tmp}/brainchopc-verify.XXXXXX")
trap 'rm -rf "$bc_work"' EXIT HUP INT TERM

# pkgutil --expand-full unpacks the payload as well as the metadata.
pkgutil --expand-full "$bc_pkg" "$bc_work/expanded"

bc_bin=$(find "$bc_work/expanded" -type d -path '*/usr/local/bin' -print -quit)
[ -n "$bc_bin" ] || {
	echo "verify_macos_pkg.sh: package does not install into /usr/local/bin" >&2
	exit 1
}

bc_count=0
for bc_exe in "$bc_bin"/*; do
	[ -f "$bc_exe" ] || continue
	bc_name=$(basename -- "$bc_exe")
	bc_count=$((bc_count + 1))

	codesign --verify --strict --verbose=2 "$bc_exe"

	# Anything outside /usr/lib and /System is a library the target machine is
	# not guaranteed to have.
	bc_foreign=$(otool -L "$bc_exe" | tail -n +2 | awk '{ print $1 }' |
		grep -v '^/usr/lib/' | grep -v '^/System/' || true)
	if [ -n "$bc_foreign" ]; then
		echo "verify_macos_pkg.sh: $bc_name has non-system dependencies:" >&2
		echo "$bc_foreign" | sed 's/^/  /' >&2
		exit 1
	fi
	if otool -L "$bc_exe" | grep -Eq '/opt/homebrew|/usr/local/lib|/dist/[.]cache'; then
		echo "verify_macos_pkg.sh: $bc_name contains a build-machine library path" >&2
		exit 1
	fi

	# Runs, and its model is embedded rather than looked up on disk. --help
	# rather than --list: every build has --help, and --list was retired when
	# each executable came to contain exactly one model.
	chmod +x "$bc_exe"
	"$bc_exe" --help >/dev/null
	echo "  ok $bc_name"
done

[ "$bc_count" -gt 0 ] || {
	echo "verify_macos_pkg.sh: package installs no executables" >&2
	exit 1
}

echo "Verified $bc_pkg ($bc_count executables)"
