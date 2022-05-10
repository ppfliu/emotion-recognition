# Group Gated Fusion on Attention-based Bidirectional Alignment for Multimodal Emotion Recognition


## Introduction
Emotion recognition is a challenging and actively-studied research area that plays a critical role in emotion-aware human-computer interaction systems.
In a multimodal setting, temporal alignment between different modalities has not been well investigated yet. This work presents a new model named as
Gated Bidirectional Alignment Network (GBAN), which consists of an attention-based bidirectional alignment network over LSTM hidden states to explicitly capture the alignment relationship
between speech and text, and a novel group gated fusion (GGF) layer to integrate the representations of different modalities.
The work has been published in INTERSPEECH-2020, please refer to the [paper](http://www.interspeech2020.org/uploadfile/pdf/Mon-1-9-6.pdf).


## Source Code
This repository contains the source code for proposed group gated fusion (GGF) layer, as well as other works such as the tensor fusion layer (TFL) from the paper "Tensor Fusion Network for Multimodal Sentiment Analysis" and the gated multimodal
units (GMU) layer from the paper "Gated Multimodal Units for Information Fusion". The code is implemented based on Tensorflow, you may use these layers directly in your model.


### How to use the layers?
```
python ggf.py # An example of how to use the group gated fusion (GGF) layer
python tfl.py # An example of how to use the tensor fusion layer (TFL) layer
python gmu.py # An example of how to use the gated multimodal units (GMU) layer
```

### How to run the experiments?
Please request the IEMOCAP dataset from https://sail.usc.edu/iemocap/ and put the dataset in the current folder.
You also need to download the GloVe vectors from https://nlp.stanford.edu/projects/glove/ and put them under the glove folder.
The batch script can be run as follows:
```
bash batch.sh
```

## Citation
If you use the released source code in your work, please cite the following paper:
<pre>
@inproceedings{liu2020group,
  title={Group Gated Fusion on Attention-Based Bidirectional Alignment for Multimodal Emotion Recognition.},
  author={Liu, Pengfei and Li, Kun and Meng, Helen},
  booktitle={INTERSPEECH},
  pages={379--383},
  year={2020}
}
</pre>


## Report
Please feel free to create an issue or send emails to the first author at ppfliu@gmail.com.
