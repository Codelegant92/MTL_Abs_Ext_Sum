
# MTL_Abs_Ext_Sum
## A Tensorflow implementation of Multi-Task Learning for Abstractive and Extractive Summarization

This repository presents a multi-task learning frame work to jointly train a sequence-to-sequence model (for abstractive summarization) and a sequence labelling model (for extractive summarization) with hierarchical attention network.

Our implementation is based on a Tensorflow implementation of seq2seq attentional model from Google AI https://ai.googleblog.com/2016/08/text-summarization-with-tensorflow.html.

## Prerequisites
+ TensorFlow
+ Bazel

# Dataset
[CNN/DailyMail](https://cs.nyu.edu/~kcho/DMQA/)

Both summarized sentences and extracted key sentences are needed as annotations.

# Before running the tran/eval/decode scripts
Install TensorFlow and Bazel.

```shell
# cd to your workspace
# 1. Clone the code to your workspace 'MTL_Abs_Ext_Sum' directory.
# 2. Create an empty 'WORKSPACE' file in your workspace.
# 3. Move the train/eval/test data to your workspace 'data' directory.
#    In the following example, I named the data cnn-dailymail-train.source, cnn-dailymail-train.target, cnn-dailymail-train.label, etc.
#    If your data files have different names, update the --data_path.

ls -R
.:
data  MTL_Abs_Ext_Sum  WORKSPACE

./data:

./MTL_Abs_Ext_Sum:
batch_reader.py       beam_search.py       BUILD    README.md      data.py           seq2seq_attention_model.py
  seq2seq_attention_decode.py  seq2seq_attention.py        seq2seq_lib_new.py   data

./MTL_Abs_Ext_Sum/data:
vocab  cnn-dailymail-train.source cnn-dailymail-train.target cnn-dailymail-train.label cnn-dailymail-test.source ...(omitted)

bazel build -c opt --config=cuda MTL_Abs_Ext_Sum/...
# Run the training.
bazel-bin/MTL_Abs_Ext_Sum/seq2seq_attention \
  --mode=train \
  --article_path=data/cnn-dailymail-train.source \
  --abstract_path=data/cnn-dailymail-train.target \
  --vocab_path=data/vocab \
  --label_path=data/cnn-dailymail-train.label \
  --log_root=log_root \
  --train_dir=log_root/train \
  ...

# Run the decode. Run it when the most is mostly converged.
bazel-bin/MTL_Abs_Ext_Sum/seq2seq_attention \
  --mode=decode \
  --article_path=data/cnn-dailymail-test.source \
  --abstract_path=data/cnn-dailymail-test.target \
  --vocab_path=data/vocab \
  --prob_path=data/cnn-dailymail-test.prob \
  --label_path=data/cnn-dailymail-test.label \
  --log_root=log_root \
  --decode_dir=log_root/decode \
  --beam_size=8
```
# Cite our paper
If our work and the code are useful to you, please cite it:
```shell
@article{chen2019multi,
  title={Multi-task learning for abstractive and extractive summarization},
  author={Chen, Yangbin and Ma, Yun and Mao, Xudong and Li, Qing},
  journal={Data Science and Engineering},
  volume={4},
  number={1},
  pages={14--23},
  year={2019},
  publisher={Springer}
}
```
