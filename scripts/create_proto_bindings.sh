#!/usr/bin/env bash

# 1) Compile all protos in this folder to *_pb2.py (flat output)
cd "../src/ecal_lib/proto_files"
protoc -I=. --python_out=. ./*.proto

# 2) Patch imports in all generated files so they resolve as package imports
#    This fixes "import header_pb2 as header__pb2" → "from ecal_lib.proto_files import header_pb2 as header__pb2"
#    and similarly for Timestamp_pb2.
for f in ./*_pb2.py; do
  # GNU sed (Linux). On macOS use: sed -i '' ...
  sed -i \
    -e 's/^import header_pb2 as /from ecal_lib.proto_files import header_pb2 as /' \
    -e 's/^import Timestamp_pb2 as /from ecal_lib.proto_files import Timestamp_pb2 as /' \
    "$f"
done
