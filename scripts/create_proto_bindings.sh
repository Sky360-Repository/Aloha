#!/bin/bash

cd ./../proto_files
protoc -I=. --python_out=. *.proto
