"""Sequence-to-Sequence with attention model for text summarization.
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import seq2seq_lib_new
from seq2seq_lib_new import attention_decoder

HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, num_labels,'
                     'enc_layers_sent, enc_layers_doc, enc_sent_num, max_enc_sent_len, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, para_lambda, para_beta, max_grad_norm, '
                     'num_softmax_samples')


def _extract_argmax_and_embed(embedding_common, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding_common: embedding tensor for source and target.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    """function that feed previous model output rather than ground truth."""
    if output_projection is not None:
      prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = tf.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding_common, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def run_train_step(self, sess, article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets,
                     article_lens_doc, article_lens_sent, abstract_lens,
                     loss_weights, article_weights_sent, article_weights_doc):
    to_return = [self._train_op, self._summaries, self._loss_total, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._enc_probs: enc_probs_batch,
                               self._enc_labels: enc_labels_batch,
                               self._targets: targets,
                               self._article_lens_doc: article_lens_doc,
                               self._article_lens_sent: article_lens_sent,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights,
                               self._article_weights_sent: article_weights_sent,
                               self._article_weights_doc: article_weights_doc})

  def run_eval_step(self, sess, article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets,
                    article_lens_doc, article_lens_sent, abstract_lens,
                    loss_weights, article_weights_sent, article_weights_doc):
    to_return = [self._summaries, self._loss_decoder, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._enc_probs: enc_probs_batch,
                               self._enc_labels: enc_labels_batch,
                               self._targets: targets,
                               self._article_lens_doc: article_lens_doc,
                               self._article_lens_sent: article_lens_sent,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights,
                               self._article_weights_sent: article_weights_sent,
                               self._article_weights_doc: article_weights_doc})

  def run_decode_step(self, sess, article_batch, abstract_batch, enc_probs_batch, enc_labels_batch, targets,
                      article_lens_doc, article_lens_sent, abstract_lens,
                      loss_weights, article_weights_sent, article_weights_doc):
    to_return = [self._outputs, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._enc_probs: enc_probs_batch,
                               self._enc_labels: enc_labels_batch,
                               self._targets: targets,
                               self._article_lens_doc: article_lens_doc,
                               self._article_lens_sent: article_lens_sent,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights,
                               self._article_weights_sent: article_weights_sent,
                               self._article_weights_doc: article_weights_doc})

  def run_tagging_step(self, sess, article_batch, abstract_batch, enc_labels_batch, targets,
                     article_lens_doc, article_lens_sent, abstract_lens,
                     loss_weights, article_weights_sent, article_weights_doc):
    to_return = [self._best_tagging, self._summaries, self._loss_total, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._enc_labels: enc_labels_batch,
                               self._targets: targets,
                               self._article_lens_doc: article_lens_doc,
                               self._article_lens_sent: article_lens_sent,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights,
                               self._article_weights_sent: article_weights_sent,
                               self._article_weights_doc: article_weights_doc})

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self._hps
    self._articles = tf.placeholder(tf.int32,
                                    [hps.batch_size, hps.enc_sent_num, None],
                                    name='articles')
    self._abstracts = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps],
                                     name='abstracts')
    self._enc_probs = tf.placeholder(tf.float32,
                                     [hps.batch_size, hps.enc_sent_num],
                                     name='enc_probs')
    self._enc_labels = tf.placeholder(tf.float32,
                                      [hps.batch_size, hps.enc_sent_num],
                                      name='enc_labels')
    self._targets = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='targets')
    self._article_lens_doc = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='article_lens_doc')
    self._article_lens_sent = tf.placeholder(tf.int32, 
                                    [hps.batch_size, hps.enc_sent_num],
                                    name='article_lens_sent')
    self._abstract_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                         name='abstract_lens')
    self._article_weights_sent = tf.placeholder(tf.float32,
                                                [hps.batch_size, hps.enc_sent_num, None],
                                                name="article_weights_sent")
    self._article_weights_doc = tf.placeholder(tf.float32,
                                               [hps.batch_size, hps.enc_sent_num])

    self._loss_weights = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                        name='loss_weights')


  def _add_seq2seq(self):
    hps = self._hps
    vsize = self._vocab.NumIds()
    #pre_trained_vocab_embedding = self._vocab.VocabEmbedding()

    with tf.variable_scope('seq2seq'):
      encoder_inputs = tf.concat(tf.unstack(self._articles, axis=1), 0)  #[(enc_sent_num*batch_size), enc_sent_len]
      encoder_inputs_weights_sent = tf.expand_dims(tf.concat(tf.unstack(self._article_weights_sent, axis=1), 0),
                                                   axis=2)  # [(enc_sent_num*batch_size), enc_sent_len, 1]
      encoder_inputs_weights_doc = self._article_weights_doc  # [batch_size, enc_sent_num]
      encoder_inputs_weights_sent_forAttn = tf.reshape(self._article_weights_sent, [hps.batch_size, -1])
      decoder_inputs = tf.unstack(tf.transpose(self._abstracts))  #dec_timesteps*[batch_size]
      targets = self._targets #[batch_size, dec_timesteps]
       #[batch_size, enc_sent_num]
      enc_probs = self._enc_probs  / (tf.reduce_sum(self._enc_probs, axis=1, keep_dims=True)+1e-12)  #[batch_size, enc_sent_num]
      enc_labels = self._enc_labels
      loss_weights = self._loss_weights  # [batch_size, dec_timesteps]
      article_lens_doc = self._article_lens_doc  # [batch_size]
      article_lens_sent = tf.concat(tf.unstack(tf.transpose(self._article_lens_sent)), 0)  # [(enc_sent_num*batch_size)]

      # Embedding shared by the input and outputs.
      with tf.variable_scope('embedding_common'), tf.device('/cpu:0'):
        '''
        embedding_common = tf.get_variable(
            'embedding_common', dtype=tf.float32,
            initializer=tf.stack(pre_trained_vocab_embedding))
        '''
        embedding_common = tf.get_variable('embedding_common', [vsize, hps.emb_dim], dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=1e-4))
        emb_encoder_inputs = tf.nn.embedding_lookup(embedding_common, encoder_inputs) #[(enc_sent_num*batch_size), enc_sent_len, emb_size]
        emb_decoder_inputs = [tf.nn.embedding_lookup(embedding_common, x)
                              for x in decoder_inputs] #dec_timesteps * [batch_size, emb_dim]

      with tf.variable_scope('encoder-sent-layer'), tf.device('/gpu:0'):
        cell_fw_sent = tf.contrib.rnn.GRUCell(
          hps.num_hidden)
        cell_bw_sent = tf.contrib.rnn.GRUCell(
          hps.num_hidden)
        (emb_encoder_inputs_tuple, _) = tf.nn.bidirectional_dynamic_rnn(
          cell_fw_sent, cell_bw_sent, emb_encoder_inputs, dtype=tf.float32,
          sequence_length=article_lens_sent)

      emb_encoder_inputs_sent = tf.concat(emb_encoder_inputs_tuple, 2) #[(enc_sent_num*batch_size), enc_sent_len, 2num_hidden]

      # encoder_inputs_doc_temp: [(enc_sent_num*batch_size), 2num_hidden]
      encoder_inputs_doc_temp1 = tf.reduce_sum(tf.multiply(emb_encoder_inputs_sent, encoder_inputs_weights_sent),
                                               axis=1) / (tf.reduce_sum(encoder_inputs_weights_sent, axis=1) + 1e-12)

      with tf.variable_scope('reform_encoder_sent'), tf.device('/gpu:0'):
        w_reform_sent = tf.get_variable('w_reform_sent', [hps.num_hidden*2, hps.num_hidden*2], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=1e-4))
        bias_reform_sent = tf.get_variable('bias_reform_sent', [hps.num_hidden*2], dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=1e-4))
        encoder_inputs_doc_temp2 = tf.tanh(tf.matmul(encoder_inputs_doc_temp1, w_reform_sent)+bias_reform_sent)

      # encoder_inputs_doc_final: [batch_size, enc_num_size, 2num_hidden]
      encoder_inputs_doc_temp3 = tf.reshape(encoder_inputs_doc_temp2, [hps.enc_sent_num, hps.batch_size, -1])

      encoder_inputs_doc = tf.reshape(tf.concat(tf.unstack(encoder_inputs_doc_temp3), axis=1),
                                      [hps.batch_size, -1, 2 * hps.num_hidden])

      with tf.variable_scope('encoder-doc-layer'), tf.device('/gpu:0'):
        cell_fw_doc = tf.contrib.rnn.GRUCell(
          hps.num_hidden)
        cell_bw_doc = tf.contrib.rnn.GRUCell(
          hps.num_hidden)
          #encoder_output_states: a fw-bw tuple of c-h tuple of tensors:
             #([batch_size, fw_h_state_size],
             # [batch_size, bw_h_state_size])
        (encoder_inputs_doc_final_tuple, encoder_output_states) = tf.nn.bidirectional_dynamic_rnn(
          cell_fw_doc, cell_bw_doc, encoder_inputs_doc, dtype=tf.float32,
          sequence_length=article_lens_doc)

      encoder_outputs = tf.concat(encoder_inputs_doc_final_tuple, 2)

      with tf.variable_scope('reform_final_state'), tf.device('/gpu:0'):
        w_reduce_h = tf.get_variable('w_reduce_h', [hps.num_hidden*2, hps.num_hidden], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-4))
        bias_reduce_h = tf.get_variable('bias_reduce_h', [hps.num_hidden], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=1e-4))
        old_h = tf.concat([encoder_output_states[0], encoder_output_states[1]], axis=1)
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)

      encoder_states_fw = new_h

      #encoder_states_fw = encoder_output_states[0]

      with tf.variable_scope('output_projection'):
        w = tf.get_variable(
            'w', [hps.num_hidden, vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        w_t = tf.transpose(w)
        v = tf.get_variable(
            'v', [vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

      with tf.variable_scope('tagging_projection'):
        '''
        w_tag = tf.get_variable(
            'w_tag', [2*hps.num_hidden, hps.num_labels], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        v_tag = tf.get_variable(
            'v_tag', [hps.num_labels], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        '''
        w_tag = tf.get_variable(
            'w_tag', [2 * hps.num_hidden, 1], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        v_tag = tf.get_variable(
            'v_tag', [1], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))


      with tf.variable_scope('decoder'), tf.device('/gpu:0'):
        # When decoding, use model output from the previous step
        # for the next step.
        loop_function = None
        if hps.mode == 'decode':
          loop_function = _extract_argmax_and_embed(
              embedding_common, (w, v), update_embedding=False)
        cell_decoder = tf.contrib.rnn.GRUCell(
            hps.num_hidden)
        self._enc_top_states = encoder_outputs  #[batch_size, enc_sent_num, 2num_hidden]

        self._enc_top_states_sent = tf.concat(
            tf.unstack(tf.reshape(emb_encoder_inputs_sent, [hps.enc_sent_num, hps.batch_size, -1, 2 * hps.num_hidden])),
            axis=1)

        self._dec_in_state = encoder_states_fw #[batch_size, fw_h_state_size]

        # During decoding, follow up _dec_in_state are fed from beam_search.
        # dec_out_state are stored by beam_search for next step feeding.
        # decoder_outputs: hps.dec_timesteps*[batch_size, hps.num_hidden]
        # self._dec_out_state: ([batch_size, cell_decoder.c_state_size], [batch_size, dell_decoder.h_state_size])
        # attention2encoder_doc: [batch_size. hps.enc_sent_num]
        initial_state_attention = (hps.mode == 'decode')
        decoder_outputs, self._dec_out_state, attention2encoder_doc, _ = attention_decoder(
            emb_decoder_inputs, self._dec_in_state, self._enc_top_states, self._enc_top_states_sent,
            cell_decoder, num_heads=1, loop_function=loop_function,
            initial_state_attention=initial_state_attention,
            encoder_inputs_weights_doc=encoder_inputs_weights_doc,
            encoder_inputs_weights_sent=encoder_inputs_weights_sent_forAttn)

      with tf.variable_scope('output'), tf.device('/gpu:0'):
        model_outputs = []
        for i in xrange(len(decoder_outputs)):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          model_outputs.append(
              tf.nn.xw_plus_b(decoder_outputs[i], w, v))

      with tf.variable_scope('tagging'), tf.device('/gpu:0'):

        encoder_tagging_inputs = tf.reshape(encoder_outputs, [-1, 2*hps.num_hidden])
        encoder_tagging_outputs_temp = tf.nn.xw_plus_b(encoder_tagging_inputs, w_tag, v_tag)
        encoder_tagging_outputs = tf.reshape(encoder_tagging_outputs_temp, [hps.batch_size, hps.enc_sent_num, -1])
        sigmoid_score = tf.nn.sigmoid(encoder_tagging_outputs)[:, :, 0]  # [batch_size, enc_sent_num]

      if hps.mode == 'decode':
        with tf.variable_scope('decode_output'):
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
          self._outputs = tf.concat(
              [tf.reshape(x, [hps.batch_size, 1]) for x in best_outputs], 1)

          self._topk_log_probs, self._topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), hps.batch_size)

      if hps.mode=='tagging':
        with tf.variable_scope('tagging_output'):
          #self._best_tagging = tf.zeros([hps.batch_size, hps.enc_sent_num], dtype=tf.float32)
          self._best_tagging = sigmoid_score

      # loss 1: sequence generation loss
      with tf.variable_scope('loss_decoder'), tf.device('/gpu:0'):
        def sampled_loss_func(labels, inputs):
          with tf.device('/gpu:0'):  # Try gpu.
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(w_t, v, labels, inputs,
                                              hps.num_softmax_samples, vsize)   #[batch_size]
        if hps.num_softmax_samples != 0 and hps.mode == 'train':
          self._loss_decoder = seq2seq_lib_new.sequence_loss(
              decoder_outputs, targets, loss_weights, softmax_loss_function=sampled_loss_func)
        else:
          model_outputs_temp = [tf.reshape(x, [hps.batch_size, 1, vsize]) for x in model_outputs]
          model_outputs_final = tf.concat(model_outputs_temp, 1)

          self._loss_decoder = tf.contrib.seq2seq.sequence_loss(
              model_outputs_final, targets, loss_weights)
        tf.summary.scalar('loss_decoder', tf.minimum(12.0, self._loss_decoder))

      # loss 2: attention distribution
      with tf.variable_scope('loss_attention'), tf.device('/gpu:0'):
        self._loss_attention = tf.nn.l2_loss((enc_probs-attention2encoder_doc)*encoder_inputs_weights_doc)/\
                               ((tf.reduce_sum(encoder_inputs_weights_doc))+1e-12)
        tf.summary.scalar('loss_attention', hps.para_lambda*self._loss_attention)

      # loss 3: sequence labeling loss
      with tf.variable_scope('loss_tagging'), tf.device('/gpu:0'):
        '''
        self._loss_tagging = tf.contrib.seq2seq.sequence_loss(
                encoder_tagging_outputs, enc_labels, encoder_inputs_weights_doc)
        '''
        self._loss_tagging = -tf.reduce_sum((enc_labels * tf.log(sigmoid_score) + (1 - enc_labels) * tf.log(1 - sigmoid_score)) * encoder_inputs_weights_doc) / \
                             (tf.reduce_sum(encoder_inputs_weights_doc)+1e-12)
        tf.summary.scalar('loss_tagging', hps.para_beta*self._loss_tagging)

      with tf.variable_scope('loss_total'), tf.device('/gpu:0'):
        self._loss_total = self._loss_decoder + hps.para_lambda*self._loss_attention + hps.para_beta*self._loss_tagging
        tf.summary.scalar('loss_total', self._loss_total)


  def _add_train_op(self):
    """Sets self._train_op, op to run for training."""
    hps = self._hps
    '''
    self._lr_rate = tf.maximum(
        hps.min_lr,  # min_lr_rate.
        tf.train.exponential_decay(hps.lr, self.global_step, 30000, 0.98))
    '''
    tvars = tf.trainable_variables()
    with tf.device('/gpu:0'):
      grads, global_norm = tf.clip_by_global_norm(tf.gradients(self._loss_total, tvars), hps.max_grad_norm)
    tf.summary.scalar('global_norm', global_norm)
    #optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
    optimizer = tf.train.AdagradOptimizer(hps.lr, initial_accumulator_value=0.1)
    #tf.summary.scalar('learning rate', self._lr_rate)
    with tf.device('/gpu:0'):
        self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')

  def encode_top_state(self, sess, enc_inputs, enc_len_doc, enc_len_sent, enc_weights_sent):
    """Return the top states from encoder for decoder.

    Args:
      sess: tensorflow session.
      enc_inputs: encoder inputs of shape [batch_size, enc_sent_num, enc_sent_len].
      enc_len_doc: encoder input doc length of shape [batch_size].
      enc_len_sent: encoder input length of shape [batch_size, enc_sent_num].
    Returns:
      enc_top_states: The top level encoder statesself.[batch_size*enc_sent_num*2num_hidden]
      dec_in_state: The decoder layer initial stateself.[batch_size, fw_h_state_size]
    """
    results = sess.run([self._enc_top_states, self._enc_top_states_sent, self._dec_in_state],
                       feed_dict={self._articles: enc_inputs,
                                  self._article_lens_doc: enc_len_doc,
                                  self._article_lens_sent: enc_len_sent,
                                  self._article_weights_sent: enc_weights_sent})
    return results[0], results[1], results[2][0]

  def decode_topk(self, sess, latest_tokens, enc_top_states,enc_top_states_sent, dec_init_states,
                  enc_weights_doc, enc_weights_sent):
    """Return the topK results and new decoder states."""

    feed = {
        self._enc_top_states: enc_top_states,
        self._enc_top_states_sent: enc_top_states_sent,
        self._dec_in_state:
            #dec_init_states,
            np.squeeze(np.array(dec_init_states)),
        self._abstracts:
            np.transpose(np.array([latest_tokens])),
        self._abstract_lens: np.ones([len(dec_init_states)], np.int32),
        self._article_weights_doc: enc_weights_doc,
        self._article_weights_sent: enc_weights_sent}

    results = sess.run(
        [self._topk_ids, self._topk_log_probs, self._dec_out_state],
        feed_dict=feed)

    ids, probs, states = results[0], results[1], results[2]
    new_states = [s for s in states]
    #new_states = states # [s for s in states] #very important!!!
    return ids, probs, new_states

  def build_graph(self):
    self._add_placeholders()
    with tf.device('/gpu:0'):
        self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
