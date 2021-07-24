"""Module for decoding."""

import os
import time

import tensorflow as tf
import beam_search
import data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decode_steps', 1000000,
                            'Number of decoding steps.')
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 4000,
                            'Number of batches to decode before restoring next '
                            'checkpoint')

DECODE_LOOP_DELAY_SECS = 60
DECODE_IO_FLUSH_INTERVAL = 100


class DecodeIO(object):
  """Writes the decoded and references to RKV files for Rouge score.

    See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
  """

  def __init__(self, outdir):
    self._cnt = 0
    self._outdir = outdir
    if not os.path.exists(self._outdir):
      os.mkdir(self._outdir)
      os.mkdir(os.path.join(self._outdir, 'reference'))
      os.mkdir(os.path.join(self._outdir, 'decoded'))

    self._ref_file = None
    self._decode_file = None

  def Write(self, reference, decode):
    """Writes the reference and decoded outputs to RKV files.

    Args:
      reference: The human (correct) result.
      decode: The machine-generated result
    """
    self.ResetFiles()

    self.saveText(reference, self._ref_file)
    self.saveText(decode, self._decode_file)
    print("Save file %d ..." % self._cnt)

    self._cnt += 1
    if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
      self._ref_file.flush()
      self._decode_file.flush()

  def ResetFiles(self):
    """Resets the output files. Must be called once before Write()."""
    if self._ref_file: self._ref_file.close()
    if self._decode_file: self._decode_file.close()
    timestamp = int(time.time())
    self._ref_file = open(
      os.path.join(os.path.join(self._outdir, 'reference'), '%d' % (self._cnt)), 'w')
    self._decode_file = open(
      os.path.join(os.path.join(self._outdir, 'decoded'), '%d' % (self._cnt)), 'w')

  def saveText(self, text, f_output):
    sentence = []
    sentence_list = []
    word_list = text.strip().split(' ')
    if (word_list[-1] != '.'):
      word_list.append('.')
    for word in word_list:
      if (word == '.'):
        sentence.append(word)
        sentence_list.append(' '.join(sentence))
        sentence = []
      else:
        sentence.append(word)
    if (len(sentence_list) > 1):
      for item in sentence_list[:-1]:
        f_output.write(item + '\n')
    f_output.write(sentence_list[-1])


class BSDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batch_reader, hps, vocab):
    """Beam search decoding.

    Args:
      model: The seq2seq attentional model.
      batch_reader: The batch data reader.
      hps: Hyperparamters.
      vocab: Vocabulary
    """
    self._model = model
    self._model.build_graph()
    self._batch_reader = batch_reader
    self._hps = hps
    self._vocab = vocab
    self._saver = tf.train.Saver()
    self._decode_io = DecodeIO(FLAGS.decode_dir)

  def DecodeLoop(self):
    """Decoding loop for long running process."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    step = 0
    while step < FLAGS.max_decode_steps:
      time.sleep(DECODE_LOOP_DELAY_SECS)
      if not self._Decode(self._saver, sess):
        continue
      step += 1

  def _Decode(self, saver, sess):
    """Restore a checkpoint and decode it.

    Args:
      saver: Tensorflow checkpoint saver.
      sess: Tensorflow session.
    Returns:
      If success, returns true, otherwise, false.
    """
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
      return False

    tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(
        FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
    tf.logging.info('renamed checkpoint path %s', ckpt_path)
    saver.restore(sess, ckpt_path)

    for _ in xrange(FLAGS.decode_batches_per_ckpt):
      (article_batch, _, article_probs_batch, _, _, article_lens_doc, article_lens_sent,
       _, _, article_weights_sent, article_weights_doc, origin_articles, origin_abstracts) = self._batch_reader.NextBatch()

      for i in xrange(self._hps.batch_size):
        bs = beam_search.BeamSearch(
            self._model, self._hps.batch_size,
            self._vocab.WordToId(data.SENTENCE_START),
            self._vocab.WordToId(data.SENTENCE_END),
            self._hps.dec_timesteps)

        #article_batch_cp has a shape [batch_size*enc_sent_num*enc_sent_len]
        #article_batch_cp[0] == article_batch_cp[1] == article_batch_size_cp[2] == ...
        article_batch_cp = article_batch.copy()
        article_batch_cp[:] = article_batch[i:i+1]
       
        article_lens_doc_cp = article_lens_doc.copy()
        article_lens_doc_cp[:] = article_lens_doc[i:i+1]
        
        article_lens_sent_cp = article_lens_sent.copy()
        article_lens_sent_cp[:] = article_lens_sent[i:i+1]

        article_weights_sent_cp = article_weights_sent.copy()
        article_weights_sent_cp[:] = article_weights_sent[i:i + 1]

        article_weights_doc_cp = article_weights_doc.copy()
        article_weights_doc_cp[:] = article_weights_doc[i:i + 1]

        best_beam = bs.BeamSearch(sess, article_batch_cp, article_lens_doc_cp, article_lens_sent_cp, article_weights_sent_cp, article_weights_doc_cp)[0]
        decode_output = [int(t) for t in best_beam.tokens[1:]]
        '''
        origin_articles_x = origin_articles[i].split(' ')
        origin_articles_i = ''.join(origin_articles_x)

        origin_abstracts_x = origin_abstracts[i].split(' ')
        origin_abstracts_i = ''.join(origin_abstracts_x)
        '''
        self._DecodeBatch(
            origin_articles[i], origin_abstracts[i], decode_output)
    return True

  def _DecodeBatch(self, article, abstract, output_ids):
    """Convert id to words and writing results.

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      output_ids: The abstract word ids output by machine.
    """
    decoded_output = ' '.join(data.Ids2Words(output_ids, self._vocab))
    end_p = decoded_output.find(data.SENTENCE_END, 0)
    if end_p != -1:
      decoded_output = decoded_output[:end_p]
    tf.logging.info('article:  %s', article)
    tf.logging.info('abstract: %s', abstract)
    tf.logging.info('decoded:  %s', decoded_output)

    self._decode_io.Write(abstract, decoded_output.strip())
