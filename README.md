# ASR project

An attempt to train simplification of `DeepSpeech2` to achieve good quality on `LibriSpeech` dataset.

We use BeamSearch with shallow fusion to obtain predictions.

Average metrics on `test-clean` set:

[checkpoint](https://drive.google.com/drive/folders/10VUp-W2u42wir_7l32k8glEBdS4nkKei?usp=sharing): `CER = 9.31, WER = 21.09`


## Installation guide

First, clone this repository to get access to the code:

  `git clone https://github.com/huwarr/asr_project.git`
  
  `cd asr_project`

Run `setup.sh` script to download requirements and all the necessary files, including checkpoint and config:

  `sh setup.sh`

Run `test.py` file to evaluate metrics and get predictions:
  ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```


## Training

Run `train.py` file with `hw_asr/configs/deepspeech_librispeech_clean.json` config:

  `python train.py --config hw_asr/configs/deepspeech_librispeech_clean.json`
