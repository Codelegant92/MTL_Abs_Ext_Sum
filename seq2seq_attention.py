
'''
This is a seq2seq model (hierarchical encoder - decoder) with
constrained hierarchical attention (word-level attention + sent-level attention).
The constraints are from the results of extractive summarization.
'''
import sys
import time
import os

import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('article_path',
                           '', 'Path expression to source articles.')
tf.app.flags.DEFINE_string('abstract_path',
                           '', 'Path expression to target abstracts.')
tf.app.flags.DEFINE_string('vocab_path',
                           '', 'Path expression to article text vocabulary file.')
tf.app.flags.DEFINE_string('prob_path',
                           '', "Path expression to source articles' probabilities for extractive summary")
tf.app.flags.DEFINE_string('label_path',
                           '', 'Path expression to labels of extractive summarization.')
tf.app.flags.DEFINE_string('emb_path',
                           '', 'Path expression to pre-trained word embedding.')
tf.app.flags.DEFINE_integer('emb_dim',
                            300, 'Dimension of word embedding.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')

tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode/tag mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('article_sentences_num', 35,
                            'Number of sentences to use from the '
                            'article.')                                                
tf.app.flags.DEFINE_integer('max_article_sentences_length', 50,
                            'Length of each sentence to use from the '
                            'article.')
tf.app.flags.DEFINE_integer('abstract_length', 100,
                            'Max number of first sentences to use from the '
                            'abstract.')                                               
tf.app.flags.DEFINE_integer('beam_size', 5,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 1000, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
           
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 20)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  sys.stdout.write('step %d, running_avg_loss: %f\n' % (step, running_avg_loss))
  return running_avg_loss

def _Train(model, data_batcher):
  """Runs model training."""
  with tf.device('/cpu:0'):
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=10)
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:
      #startTime0 = time.time()
      #startTime1 = time.time()
      (article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets, article_lens_doc, article_lens_sent, abstract_lens,
      loss_weights, enc_weights_sent, enc_weights_doc, _, _) = data_batcher.NextBatch()
      #endTime1 = time.time()
      #print "read data using %f secs"%(endTime1-startTime1)
      #startTime2 = time.time()
      (_, summaries, loss, train_step) = model.run_train_step(
          sess, article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets, article_lens_doc, article_lens_sent,
          abstract_lens, loss_weights, enc_weights_sent, enc_weights_doc)
      #endTime2 = time.time()
      #print "run train step using %f secs"%(endTime2-startTime2)
      summary_writer.add_summary(summaries, train_step)
      running_avg_loss = _RunningAvgLoss(
          running_avg_loss, loss, summary_writer, train_step)

      step += 1
      if step % 100 == 0:
        summary_writer.flush()
      #endTime0 = time.time()
      #print "each batch using %f secs"%(endTime0-startTime0)
    sv.Stop()


def _Eval(model, data_batcher, vocab=None):              
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  bestmodel_save_path = os.path.join(FLAGS.eval_dir, 'bestmodel')
  best_loss = None

  while True:
    time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    (article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets, article_lens_doc, article_lens_sent, abstract_lens,
     loss_weights, enc_weights_sent, enc_weights_doc, _, _) = data_batcher.NextBatch()
    (summaries, loss, train_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets, article_lens_doc, article_lens_sent,
        abstract_lens, loss_weights, enc_weights_sent, enc_weights_doc)
    '''
    tf.logging.info(
        'article:  %s',
        ' '.join(data.Ids2Words(article_batch[0][0][:].tolist(), vocab)))      
    tf.logging.info(
        'abstract: %s',
        ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))    
    '''
    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _RunningAvgLoss(
        running_avg_loss, loss, summary_writer, train_step)

    if best_loss is None or running_avg_loss < best_loss:
        tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
        saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
        best_loss = running_avg_loss

    step += 1
    if step % 100 == 0:
      summary_writer.flush()

def _Tag(model, data_batcher, hps, vocab):
  "Runs model tagging"
  model.build_graph()
  saver = tf.train.Saver(max_to_keep=10)
  summary_writer = tf.summary.FileWriter(FLAGS.tagging_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  while step<230:
    print("Step: %d" % step)
    time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    (article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets, article_lens_doc, article_lens_sent,
     abstract_lens, loss_weights, enc_weights_sent, enc_weights_doc, origin_articles, origin_abstracts) = data_batcher.NextBatch()
    (sigmoid_score, summaries, loss, train_step) = model.run_tagging_step(
    sess, article_batch, abstract_batch, enc_labels_batch, targets, article_lens_doc,
    article_lens_sent, abstract_lens, loss_weights, enc_weights_sent, enc_weights_doc)
    if not os.path.exists(FLAGS.tagging_dir):
      os.mkdir(FLAGS.tagging_dir)

    tag_file = open(os.path.join(FLAGS.tagging_dir, 'tag'), 'a')
    ref_file = open(os.path.join(FLAGS.tagging_dir, 'ref'), 'a')

    gen_abstract_file = open(os.path.join(FLAGS.tagging_dir, 'gen_abstract'), 'a')
    ref_abstract_file = open(os.path.join(FLAGS.tagging_dir, 'ref_abstract'), 'a')
    origin_abstract_file = open(os.path.join(FLAGS.tagging_dir, 'origin_abstract'), 'a')

    acc = 0
    best_tagging = np.zeros([hps.batch_size, hps.enc_sent_num], dtype=np.float32)
    for i in range(hps.batch_size):
      tagging_list = []
      ref_list = []
      tagging = []
      ref = []

      for j in range(article_lens_doc[i]):
        if(sigmoid_score[i][j] >= 0.5):
          tagging_list.append('1.0')
          tagging.append(1.0)
          gen_abstract_file.write(origin_articles[i][j] + ' ')
        else:
          tagging_list.append('0.0')
          tagging.append(0.0)

        ref_list.append(str(enc_labels_batch[i][j]))

        ref.append(float(enc_labels_batch[i][j]))

        if(enc_labels_batch[i][j] == 1.0):
          ref_abstract_file.write(origin_articles[i][j]+' ')
      acc += list(np.array(tagging) - np.array(ref)).count(0.0)*1.0 / len(tagging)

      tag_file.write(' '.join(tagging_list) + '\n')
      ref_file.write(' '.join(ref_list) + '\n')
      gen_abstract_file.write('\n')
      ref_abstract_file.write('\n')
      origin_abstract_file.write(origin_abstracts[i] + '\n')

    origin_abstract_file.close()
    ref_abstract_file.close()
    gen_abstract_file.close()
    ref_file.close()
    tag_file.close()

    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _RunningAvgLoss(running_avg_loss, loss, summary_writer, train_step)
    print "average accuracy: %f" % (acc*1.0/hps.batch_size)
    step += 1
    if step % 100 == 0:
      summary_writer.flush()

def main(unused_argv):
    
  vocab = data.Vocab(FLAGS.vocab_path, FLAGS.emb_path, FLAGS.emb_dim, 50000)
  # Check for presence of required special tokens.
  assert vocab.CheckVocab(data.PAD_TOKEN) > 0
  assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
  assert vocab.CheckVocab(data.SENTENCE_START) > 0
  assert vocab.CheckVocab(data.SENTENCE_END) > 0

  batch_size = 16
  if FLAGS.mode == 'decode':
    batch_size = FLAGS.beam_size

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=batch_size,
      num_labels=2,
      enc_layers_sent=1,
      enc_layers_doc=1,
      enc_sent_num=FLAGS.article_sentences_num,
      max_enc_sent_len=FLAGS.max_article_sentences_length,
      dec_timesteps=FLAGS.abstract_length,
      min_input_len=1,  # discard articles/summaries < than this
      num_hidden=200,  # for rnn cell
      emb_dim=FLAGS.emb_dim,  # If 0, don't use embedding
      para_lambda=100,    # the weight for attention loss
      para_beta=0.5,
      max_grad_norm=2,
      num_softmax_samples=0)  # If 0, no sampled softmax.

  print "initialize batcher......"
  batcher = batch_reader.Batcher(
      FLAGS.article_path, FLAGS.abstract_path, FLAGS.prob_path, FLAGS.label_path, vocab, hps, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input)                                 
  tf.set_random_seed(FLAGS.random_seed)

  print "read all batches......"
  batcher._MakeBatchesWithBuckets()
  batcher.CreateNewBuckets()

  if hps.mode == 'train':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab)
    _Train(model, batcher)                                                          

  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab)
    _Eval(model, batcher, vocab=vocab)

  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab)
    decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
    decoder.DecodeLoop()

  elif hps.mode == 'tagging':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
          hps, vocab)
    _Tag(model, batcher, hps, vocab=vocab)

if __name__ == '__main__':
  tf.app.run()