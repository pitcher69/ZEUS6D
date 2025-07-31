#!/usr/bin/env bash

USER="pitcher69"
REPO="IITISOC"
BRANCH="main"
BASE_DATASET="DATA/mustard/video"
SUBFOLDERS=(rgb mask depth)

TMP_CLONE="tmp_repo_clone"
TARGET_DIR="data"

echo cleanup

rm -rf "$TMP_CLONE"
echo clone

git clone --depth=1 --branch "$BRANCH" "https://github.com/$USER/$REPO.git" "$TMP_CLONE"

echo copying files
for SUB in "${SUBFOLDERS[@]}"
do
    SRC_PATH="$TMP_CLONE/$BASE_DATASET/$SUB"
    DEST_PATH="$TARGET_DIR/$SUB"
    mkdir -p "$DEST_PATH"

    if [ -d "$SRC_PATH" ]; then
        find "$SRC_PATH" -maxdepth 1 -type f -iname '*.png' -exec cp {} "$DEST_PATH" \;
        echo "Copied PNG files from $SRC_PATH to $DEST_PATH"
    else
        echo "Warning: $SRC_PATH does not exist."
    fi
done

cp fused_query_gedi.npy $TARGET_DIR 

echo final cleanup
rm -rf "$TMP_CLONE"
