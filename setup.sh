#!/bin/env bash
pip install -r requirements.txt

mkdir lm
cd lm
wget "http://openslr.elda.org/resources/11/3-gram.pruned.3e-7.arpa.gz"
wget "http://www.openslr.org/resources/11/librispeech-vocab.txt"
cd ..

mkdir default_test_model
cd default_test_model
pip install --upgrade --no-cache-dir gdown
gdown https://drive.google.com/uc?id=1RKZ3ywb9EbRTrmGNWr51pp0e_eVhEwPg
gdown https://drive.google.com/uc?id=1yuaNBEoA-Uqfddc1RXIe8VmoKtTo0usG
cd ,,