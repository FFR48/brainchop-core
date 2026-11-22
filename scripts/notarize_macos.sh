#!/bin/sh
# Notarise and staple a signed installer package.
#
#   notarize_macos.sh FILE.pkg [KEYCHAIN_PROFILE]
set -eu

bc_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bc_pkg=${1:?usage: notarize_macos.sh FILE.pkg [KEYCHAIN_PROFILE]}
bc_profile=${2:-brainchopc-notary}

[ -f "$bc_pkg" ] || {
	echo "notarize_macos.sh: file not found: $bc_pkg" >&2
	exit 2
}

# An unsigned package reaches notarytool and fails there with a much less
# obvious message, so reject it here. Packages are signed with a Developer ID
# Installer certificate, which pkgutil checks -- codesign does not verify
# package signatures.
pkgutil --check-signature "$bc_pkg" >/dev/null 2>&1 || {
	echo "notarize_macos.sh: $bc_pkg is unsigned or its signature is invalid." >&2
	echo "  Set MACOS_INSTALLER_IDENTITY and rebuild with 'make macos-pkg'." >&2
	exit 2
}

xcrun notarytool submit "$bc_pkg" --keychain-profile "$bc_profile" --wait
xcrun stapler staple "$bc_pkg"
xcrun stapler validate "$bc_pkg"
"$bc_root/scripts/verify_macos_pkg.sh" "$bc_pkg"
# Installer packages are assessed under the "install" policy; "open" is for
# applications and would report a spurious rejection here.
spctl --assess --type install --verbose=4 "$bc_pkg"
shasum -a 256 "$bc_pkg"
echo "Notarized, stapled, and Gatekeeper-validated $bc_pkg"
