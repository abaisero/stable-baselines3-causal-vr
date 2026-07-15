#!/usr/bin/env bash
target=$(awk -F',' -v s="$1" 'NR>1 && $1==s {print $2 ":" $3}' servers.csv)
[[ -n "$target" ]] || { echo "unknown server: $1" >&2; exit 1; }
echo "$target"
