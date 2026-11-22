#!/bin/sh
set -eu

bc_profile=${1:?usage: store_notary_credentials.sh PROFILE APPLE_ID TEAM_ID}
bc_apple_id=${2:?usage: store_notary_credentials.sh PROFILE APPLE_ID TEAM_ID}
bc_team_id=${3:?usage: store_notary_credentials.sh PROFILE APPLE_ID TEAM_ID}

echo "Enter the app-specific password at Apple's secure prompt. It will be stored in Keychain."
echo "Do not pass the password as a command-line argument or Make variable."
xcrun notarytool store-credentials "$bc_profile" \
	--apple-id "$bc_apple_id" \
	--team-id "$bc_team_id"
