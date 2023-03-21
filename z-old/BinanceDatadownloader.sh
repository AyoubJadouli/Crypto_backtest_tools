#!/bin/bash



pair=BTCUSDT   # replace with your desired pair
output_dir=./database/Trades/$pair

# create output directory if it doesn't exist
mkdir -p $output_dir

# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# sudo ./aws/install

# download all data files for the pair
aws s3 sync --no-sign-request s3://data.binance.vision/$pair/ $output_dir --exclude "*" --include "*.zip"

echo "Done downloading data for $pair"
