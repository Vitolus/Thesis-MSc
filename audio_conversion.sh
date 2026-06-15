#!/bin/bash

mkdir -p "./data/synthesized_train_16k"
echo "Converting audio to 16kHz mono..."
find "./data/synthesized_train" -name "*.wav" | xargs -P 8 -I {} sh -c 'ffmpeg -y -i "{}" -ar 16000 -ac 1 -loglevel error "./data/synthesized_train_16k/$(basename "{}")"'
echo "Conversion complete!"


mkdir -p "./data/synthesized_test_16k"
echo "Converting audio to 16kHz mono..."
find "./data/synthesized_test" -name "*.wav" | xargs -P 8 -I {} sh -c 'ffmpeg -y -i "{}" -ar 16000 -ac 1 -loglevel error "./data/synthesized_test_16k/$(basename "{}")"'
echo "Conversion complete!"