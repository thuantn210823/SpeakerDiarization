# SpeakerDiarization
This repository addresses the Speaker Diarization problem in an end-to-end manner. For future research and comparison, I have reimplemented several popular **EEND** models, covering everything from creating simulated datasets to building and evaluating the models. The models studied include **SA-EEND** from '*End-to-end neural speaker diarization with self-attention*', **EEND-EDA** from '*End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors*', and **EEND-VC** from '*Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds*', which tackle the challenges of overlapping speech, an unknown number of speakers, and long recordings in Speaker Diarization task.
## Installation
Clone my repo
```bash
$ git clone https://github.com/thuantn210823/SpeakerDiarization.git
```
Install all required libraries in the `requirements.txt` file.
```bash
cd SpeakerDiarization
pip install -r requirements.txt
```
## Simulated Dataset
To address the scarcity of annotated data, simulated datasets allow us to pretrain models and then adapt them to specific datasets. In this work, I followed the **Simulation Conversation** algorithm from *From Simulated Mixtures to Simulated Conversations as Training Data for End-to-End Neural Diarization* paper. For more details, check the file `make_mixtures.py` and my [kaggle notebook](https://www.kaggle.com/code/ngcthun/d-sd-3-dilibrisc4-test-dev) if you interest.  

For data preparation, you can find in the `Diarization_dataset.py` file, or refer to the [original authors' repo](https://github.com/hitachi-speech/EEND).

## Run
For training
```sh
cd SpeakerDiarization
py train.py --config_yaml YAML_PATH
```
For inference
```sh
cd SpeakerDiarization
py infer.py --config_yaml YAML_PATH --audio_path AUDIO_PATH
```
`Note:` If the above command doesn’t work, try replacing `py` with `python`, or the full `python.exe` path (i.e. `~/Python3xx/python.exe`) if the above code doesn't work.
## Example
```sh
cd SpeakerDiarization
py train.py --config_yaml conf/EEND_VC/train.yaml
```
```sh
cd SpeakerDiarization
py infer.py --config_yaml conf/EEND_VC/infer.yaml --audio_path example/1089-121-2-2.wav
```
`Note:` Some arguments in these `train.yaml` files are still left blank waiting for you to complete. 

You should find the file `pred_1089-121-2-2.rttm` within your cloned repository for the above inference. To modify other settings, such as chunk size, you can edit the `infer.yaml` file located in the `conf` directory.
## Pretrained Models
Pretrained models are offerred here, which you can find in the `pretrained_models` directory. 

## Results
All models were evaluated using the publicly available CALLHOME American English dataset from Talkbank. You can access it here: [talkbank/callhome](https://huggingface.co/datasets/talkbank/callhome). Since the dataset does not provide separate validation and test sets, I randomly split it into two parts using seed 42. The first part was used for the domain adaptation step, while the results below are from the second part, which served as the test set. With only 1000 hours of training data, my results may be slightly worse.

I also tested servel SOTA methods for fair comparision. The first model is **SA-EEND** from Xflick, a PyTorch implementation of the original model trained on the Switchboard Phase 1 dataset, yielding results very similar to the published ones. And the other model is **Pyannote 3.1**, one of the top state-of-the-art models at the time.
| Model | Adapted | Chunking | Clustering Method   | #DER  | #MI  | #FA  | #CF  |
|:-----:|:-------:|:--------:|:-------------------:|:-----:|:----:|:----:|:----:|
| [Pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | x | -        | -                   | 25.26% | 10.42% | 5.56% | 6.28% |
| [SA-EEND](https://github.com/Xflick/EEND_PyTorch) | v | x        | N/A                 | 18.34% | 11.06% | 4.49% | 2.79% |
| ---------- | ------- | -------- | ------------------- | ------- | ------ | ----- | ----- | <!-- Simulate double line -->
| SA-EEND | v | x        | N/A                 | 20.51% | 11.69% | 4.89% | 3.93% |
| EEND-EDA | v | x        | N/A                 | 17.69% | 9.13%  | 5.86% | 2.69% |
| EEND-VC  | v | x        | x                   | 21.95% | 12.06% | 4.46% | 5.43% |
| EEND-VC  | v | v        | Constrained-AHC     | 23.68% | 12.05% | 5.27% | 6.37% | 

Another test evaluated was the [AMI Copus](https://groups.inf.ed.ac.uk/ami/download/), with all setups following the [AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup). My baseline is **DiaPer** model.
| Model | #DER  |
|:-----:|:-------:|
| [DiaPer](https://github.com/BUTSpeechFIT/DiaPer) | 30.49% | 
|---------|-----|
| EEND-VC  | 41.29% | 

## Citation
Cite their great papers!
```
@inproceedings{fujita2019end,
  title={End-to-end neural speaker diarization with self-attention},
  author={Fujita, Yusuke and Kanda, Naoyuki and Horiguchi, Shota and Xue, Yawen and Nagamatsu, Kenji and Watanabe, Shinji},
  booktitle={2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={296--303},
  year={2019},
  organization={IEEE}
}
```
```
@article{horiguchi2020end,
  title={End-to-end speaker diarization for an unknown number of speakers with encoder-decoder based attractors},
  author={Horiguchi, Shota and Fujita, Yusuke and Watanabe, Shinji and Xue, Yawen and Nagamatsu, Kenji},
  journal={arXiv preprint arXiv:2005.09921},
  year={2020}
}
```
```
@inproceedings{kinoshita2021integrating,
  title={Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds},
  author={Kinoshita, Keisuke and Delcroix, Marc and Tawara, Naohiro},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7198--7202},
  year={2021},
  organization={IEEE}
}
```
```
@article{kinoshita2021advances,
  title={Advances in integration of end-to-end neural and clustering-based diarization for real conversational speech},
  author={Kinoshita, Keisuke and Delcroix, Marc and Tawara, Naohiro},
  journal={arXiv preprint arXiv:2105.09040},
  year={2021}
}
```
```
@article{landini2022simulated,
  title={From simulated mixtures to simulated conversations as training data for end-to-end neural diarization},
  author={Landini, Federico and Lozano-Diez, Alicia and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  journal={arXiv preprint arXiv:2204.00890},
  year={2022}
}
```
```
@article{park2022review,
  title={A review of speaker diarization: Recent advances with deep learning},
  author={Park, Tae Jin and Kanda, Naoyuki and Dimitriadis, Dimitrios and Han, Kyu J and Watanabe, Shinji and Narayanan, Shrikanth},
  journal={Computer Speech \& Language},
  volume={72},
  pages={101317},
  year={2022},
  publisher={Elsevier}
}
```
