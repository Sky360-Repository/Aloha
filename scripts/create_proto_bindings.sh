#!/bin/bash

# Change directory to the proto files location first
cd "../src/ecal/proto_files"
protoc -I=. --python_out=. *.proto
