"""Batch reader to seq2seq attention model, with bucketing support."""

from collections import namedtuple
from random import shuffle
import time

import numpy as np
import tensorflow as tf

import data

ModelInput = namedtuple('ModelInput',
                        'enc_inputs dec_inputs enc_probs enc_labels targets enc_len_doc enc_len_sent dec_len '
                        'origin_articles origin_abstracts')


class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self, article_path, abstract_path, prob_path, label_path, vocab, hps, bucketing=True, truncate_input=False):                                 ###
    """Batcher constructor.

    Args:
      article_path: article file path.
      abstract_path: abstract file path.
      label_path: label file path.
      vocab: Vocabulary.
      hps: Seq2SeqAttention model hyperparameters.
      bucketing: Whether bucket articles of similar length into the same batch.
      truncate_input: Whether truncate articles or abstracts which are too long.
    """
    self._article_path = article_path
    self._abstract_path = abstract_path
    self._prob_path = prob_path
    self._label_path = label_path
    self._vocab = vocab
    self._hps = hps
    self._bucketing = bucketing
    self._truncate_input = truncate_input

  def NextBatch(self):
    """Returns a batch of inputs for seq2seq attention model.
    Returns:
      enc_batch: A batch of encoder inputs [batch_size, hps.enc_sent_num, hps.enc_sent_len].
      dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].
      target_batch: A batch of targets [batch_size, hps.dec_timestamps].
      enc_input_len_doc: encoder input lengths (sentence number) of the batch [batch_size].
      enc_input_lens_sent: encoder input lengths (sentence lengths) of the batch [batch_size, hps.enc_sent_num].
      dec_input_len: decoder input lengths of the batch.
      loss_weights: weights for loss function, 1 if not padded, 0 if padded.
      origin_articles: original article words.
      origin_abstracts: original abstract words.
    """

    enc_pad_id = self._vocab.WordToId(data.PAD_TOKEN)

    try:
      buckets = self._buckets.next()
    except StopIteration:
      self.CreateNewBuckets()
      buckets = self._buckets.next()
      print "Start a new epoch ......"

    max_enc_batch_sent_length = max([max(buckets[i][6]) for i in range(self._hps.batch_size)])

    enc_batch = np.zeros(
        (self._hps.batch_size, self._hps.enc_sent_num, max_enc_batch_sent_length), dtype=np.int32)
    enc_input_lens_doc = np.zeros(
        (self._hps.batch_size), dtype=np.int32)    
    enc_input_lens_sent = np.zeros(
        (self._hps.batch_size, self._hps.enc_sent_num), dtype=np.int32)
    enc_probs_batch = np.zeros(
        (self._hps.batch_size, self._hps.enc_sent_num), dtype=np.float32)
    enc_labels_batch = np.zeros(
        (self._hps.batch_size, self._hps.enc_sent_num), dtype=np.float32)
    dec_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)    
    dec_output_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)   
    target_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)

    #loss1: decoder cross-entropy loss
    loss_weights = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.float32)
    #loss2: sequence labeling cross_entropy loss
    enc_weights_sent = np.zeros(
      (self._hps.batch_size, self._hps.enc_sent_num, max_enc_batch_sent_length), dtype=np.float32)
    enc_weights_doc = np.zeros(
      (self._hps.batch_size, self._hps.enc_sent_num), dtype=np.float32)
    origin_articles = ['None'] * self._hps.batch_size
    origin_abstracts = ['None'] * self._hps.batch_size

    for i in xrange(self._hps.batch_size):
      (enc_inputs, dec_inputs, enc_probs, enc_labels, targets, enc_len_doc, enc_len_sent, dec_len,
       article, abstract) = buckets[i]
      # Pad for encoding
      for j in xrange(len(enc_inputs)):
        while len(enc_inputs[j]) < max_enc_batch_sent_length:
          enc_inputs[j].append(enc_pad_id)
      while len(enc_inputs) < self._hps.enc_sent_num:
        enc_inputs.append([enc_pad_id] * max_enc_batch_sent_length)
        enc_len_sent.append(0)
        enc_probs.append(0)
        enc_labels.append(0)

      origin_articles[i] = article
      origin_abstracts[i] = abstract
      enc_input_lens_doc[i] = enc_len_doc
      enc_input_lens_sent[i, :] = enc_len_sent[:]
      enc_probs_batch[i] = enc_probs
      enc_labels_batch[i] = enc_labels
      dec_output_lens[i] = dec_len
      enc_batch[i, :, :] = np.array(enc_inputs)
      dec_batch[i, :] = dec_inputs[:]
      target_batch[i, :] = targets[:]

      for k in xrange(len(enc_len_sent)):
        if(enc_len_sent[k] != 0):
          for l in xrange(enc_len_sent[k]):
            enc_weights_sent[i][k][l] = 1
      for h in xrange(enc_len_doc):
        enc_weights_doc[i][h] = 1
      for m in xrange(dec_len):
        loss_weights[i][m] = 1

    return (enc_batch, dec_batch, enc_probs_batch, enc_labels_batch, target_batch, enc_input_lens_doc, enc_input_lens_sent, dec_output_lens,
            loss_weights, enc_weights_sent, enc_weights_doc, origin_articles, origin_abstracts)

  def _ReadFromFiles(self):
    """Fill input queue with ModelInput."""
    print "Read from files......"
    dec_start_id = self._vocab.WordToId(data.SENTENCE_START)
    dec_end_id = self._vocab.WordToId(data.SENTENCE_END)
  
    enc_input_gen = data.ExampleGen(self._article_path, num_epochs=1)
    dec_input_gen = data.ExampleGen(self._abstract_path, num_epochs=1)
    prob_gen = data.ExampleGen(self._prob_path, num_epochs=1)
    label_gen = data.ExampleGen(self._label_path, num_epochs=1)

    x = 1
    while True:
      if(x%1000 == 0):
        print "News " + str(x) + " ......"
      x += 1
      article = enc_input_gen.next()
      abstract = dec_input_gen.next()
      prob_str = prob_gen.next()
      label_str = label_gen.next()

      article_sentences = [sent.strip() for sent in data.ToSentences(article, include_token=False)]
      abstract_sentences = [sent.strip() for sent in data.ToSentences(abstract, include_token=False)]
      prob_float = [float(item) for item in prob_str.split()]
      label_float = [float(item) for item in label_str.split()]

      enc_inputs = []
      enc_probs = []
      enc_labels = []

      # Use the <s> as the <GO> symbol for decoder inputs.
      dec_inputs = [dec_start_id]

      # Convert first N sentences to word IDs, stripping existing <s> and </s>.
      for i in xrange(min(self._hps.enc_sent_num, len(article_sentences))):
        enc_inputs.append(data.GetWordIds(article_sentences[i], self._vocab))
        if(prob_float[i]==0.5):
          enc_probs.append(0)
        else:
          enc_probs.append(prob_float[i])

        if(label_float[i]==2.0):
          enc_labels.append(0.0)
        else:
          enc_labels.append(label_float[i])

      for i in xrange(len(abstract_sentences)):
        dec_inputs += data.GetWordIds(abstract_sentences[i], self._vocab)
     
      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        if (len(dec_inputs) > self._hps.dec_timesteps):
          tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d',
                             len(enc_inputs), len(dec_inputs))
          continue
      # If we are truncating input, do so if necessary
      else:
        for i in range(len(enc_inputs)):
          if len(enc_inputs[i]) > self._hps.max_enc_sent_len:
            enc_inputs[i] = enc_inputs[i][:self._hps.max_enc_sent_len]
        if len(dec_inputs) > self._hps.dec_timesteps:
          dec_inputs = dec_inputs[:self._hps.dec_timesteps]
      
      # targets is dec_inputs without <s> at beginning, plus </s> at end
      targets = dec_inputs[1:]
      targets.append(dec_end_id)

      # Now len(enc_inputs) should be <= enc_timesteps, and
      # len(targets) = len(dec_inputs) should be <= dec_timesteps
      enc_len_doc = len(enc_inputs)
      enc_len_sent = []
      for i in xrange(enc_len_doc):
        enc_len_sent.append(len(enc_inputs[i]))
      dec_len = len(targets)

      if (enc_len_doc < self._hps.min_input_len or
          sum(enc_len_sent) < 10 or
          len(dec_inputs) < self._hps.min_input_len):
        tf.logging.warning('Drop an example - too short. \nenc:%d sentences; %d words\ndec:%d',
                           len(enc_inputs), sum(enc_len_sent), len(dec_inputs))
        print("[Article]: "+' '.join(article_sentences))
        print("[Abstract]: "+' '.join(abstract_sentences))
        continue
      '''
      if (self._hps.mode=="train" and sum(enc_len_sent) < len(dec_inputs)):
        tf.logging.warning('Drop an example - article is shorter than abstract. \nenc:%d sentences; %d words\ndec:%d',
                           len(enc_inputs), sum(enc_len_sent), len(dec_inputs))
        print("[Article]: " + ' '.join(article_sentences))
        print("[Abstract]: " + ' '.join(abstract_sentences))
        continue
      '''
      # Pad if necessary
      while len(dec_inputs) < self._hps.dec_timesteps:
        dec_inputs.append(dec_end_id)
      while len(targets) < self._hps.dec_timesteps:
        targets.append(dec_end_id)     

      element = ModelInput(enc_inputs, dec_inputs, enc_probs, enc_labels, targets, enc_len_doc,
                           enc_len_sent, dec_len, ' '.join(article_sentences),
                           ' '.join(abstract_sentences))
      yield element

  def _MakeBatchesWithBuckets(self):
    """Fill bucketed batches into the bucket_input_queue."""
    print "Make batches with buckets......"
    self._inputs = []
    print "save elements ......"
    for element in self._ReadFromFiles():
      self._inputs.append(element)

  def ReadAllBatches(self):
    if self._hps.mode == "train":
      if self._bucketing:
        self._inputs = sorted(self._inputs, key=lambda inp: max(inp[6]))
      else:
        shuffle(self._inputs)
    batches = []
    print "Save batches ......"
    for i in xrange(0, len(self._inputs) - self._hps.batch_size, self._hps.batch_size):
      batches.append(self._inputs[i:i + self._hps.batch_size])
    batches.append(self._inputs[(len(self._inputs) - self._hps.batch_size):])
    for b in batches:
      yield b

  def CreateNewBuckets(self):
    self._buckets = self.ReadAllBatches()
