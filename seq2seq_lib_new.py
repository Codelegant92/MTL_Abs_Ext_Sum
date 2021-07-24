from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# We disable pylint because we need python3 compatibility.

from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function

def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states_doc,
                      attention_states_sent,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False,
                      encoder_inputs_weights_doc=None,
                      encoder_inputs_weights_sent=None):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states_doc.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states_doc must be known: %s" %
                     attention_states_doc.get_shape())
  if attention_states_sent.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states_sent must be known: %s" %
                     attention_states_sent.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.

    attn_length_doc = attention_states_doc.get_shape()[1].value
    if attn_length_doc is None:
      attn_length_doc = array_ops.shape(attention_states_doc)[1]
    attn_size_doc = attention_states_doc.get_shape()[2].value
    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden_doc = array_ops.reshape(attention_states_doc,
                               [-1, attn_length_doc, 1, attn_size_doc])
    hidden_features_doc = []
    v_doc = []
    attention_vec_size_doc = attn_size_doc  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k_doc = variable_scope.get_variable("AttnW_doc_%d" % a,
                                      [1, 1, attn_size_doc, attention_vec_size_doc])
      hidden_features_doc.append(nn_ops.conv2d(hidden_doc, k_doc, [1, 1, 1, 1], "SAME"))
      v_doc.append(
          variable_scope.get_variable("AttnV_doc_%d" % a, [attention_vec_size_doc]))

    attn_length_sent = attention_states_sent.get_shape()[1].value
    if attn_length_sent is None:
      attn_length_sent = array_ops.shape(attention_states_sent)[1]
    attn_size_sent = attention_states_sent.get_shape()[2].value
    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden_sent = array_ops.reshape(attention_states_sent,
                                   [-1, attn_length_sent, 1, attn_size_sent])
    hidden_features_sent = []
    v_sent = []
    attention_vec_size_sent = attn_size_sent  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k_sent = variable_scope.get_variable("AttnW_sent_%d" % a,
                                          [1, 1, attn_size_sent, attention_vec_size_sent])
      hidden_features_sent.append(nn_ops.conv2d(hidden_sent, k_sent, [1, 1, 1, 1], "SAME"))
      v_sent.append(
        variable_scope.get_variable("AttnV_sent_%d" % a, [attention_vec_size_sent]))

    state = initial_state

    def softmax_with_weights(X, weights):
      X_c = X-math_ops.reduce_max(X, axis=1, keep_dims=True)
      X_c_exp = math_ops.exp(X_c)
      x_softmax = (X_c_exp * weights) / math_ops.reduce_sum(X_c_exp*weights, axis=1, keep_dims=True)
      return x_softmax

    def softmax_with_weights_sent(X, weights, attn_doc_para):
      X_reshape = array_ops.reshape(X, [batch_size, attn_length_doc, -1])
      weights_reshape = array_ops.reshape(weights, [batch_size, attn_length_doc, -1])
      X_s = X_reshape-math_ops.reduce_max(X_reshape, axis=2, keep_dims=True)
      X_s_exp = math_ops.exp(X_s)
      x_s_softmax_temp = (X_s_exp * weights_reshape) / (math_ops.reduce_sum(X_s_exp * weights_reshape, axis=2, keep_dims=True)+1e-12)
      x_s_softmax = tf.reshape(x_s_softmax_temp * tf.expand_dims(attn_doc_para, axis=2), [batch_size, -1])
      return x_s_softmax

    def attention_doc(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds_doc = []  # Results of attention reads will be stored here.
      attentions_doc = [] # Values of attention will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.(transforn the tuple into list)
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1) # [batch_size, (c_state_size+h_state_size)]
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_doc_%d" % a):
          y_doc = linear(query, attention_vec_size_doc, True) #[batch_size, attention_vec_size]
          y_doc = array_ops.reshape(y_doc, [-1, 1, 1, attention_vec_size_doc])
          # Attention mask is a softmax of v^T * tanh(...). [batch_size, attn_length]
          s_doc = math_ops.reduce_sum(v_doc[a] * math_ops.tanh(hidden_features_doc[a] + y_doc),
                                  [2, 3])
          #attn = nn_ops.softmax(s)
          attn_doc = softmax_with_weights(s_doc, encoder_inputs_weights_doc)
          attentions_doc.append(attn_doc)
          # Now calculate the attention-weighted vector d.
          d_doc = math_ops.reduce_sum(
              array_ops.reshape(attn_doc, [-1, attn_length_doc, 1, 1]) * hidden_doc, [1, 2])
          ds_doc.append(array_ops.reshape(d_doc, [-1, attn_size_doc]))
      return ds_doc, math_ops.reduce_mean(array_ops.stack(attentions_doc), axis=0)

    def attention_sent(query, attn_doc_para):
      ds_sent = []
      attentions_sent = []
      if nest.is_sequence(query):
        query_list = nest.flatten(query)
        for q in query_list:
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_sent_%d" % a):
          y_sent = linear(query, attention_vec_size_sent, True)
          y_sent = array_ops.reshape(y_sent, [-1, 1, 1, attention_vec_size_sent])
          s_sent = math_ops.reduce_sum(v_sent[a] * math_ops.tanh(hidden_features_sent[a] + y_sent),
                                       [2, 3])
          # [batch_size, attn_length_doc*attn_length_sent]
          attn_sent = softmax_with_weights_sent(s_sent, encoder_inputs_weights_sent, attn_doc_para)
          attentions_sent.append(attn_sent)
          d_sent = math_ops.reduce_sum(
            array_ops.reshape(attn_sent, [-1, attn_length_sent, 1, 1]) * hidden_sent, [1, 2])
          ds_sent.append(array_ops.reshape(d_sent, [-1, attn_size_sent]))
      return ds_sent, math_ops.reduce_mean(array_ops.stack(attentions_sent), axis=0)

    outputs = []
    attention2encoder_doc = []
    attention2encoder_sent = []

    prev = None
    batch_attn_size_doc = array_ops.stack([batch_size, attn_size_doc])
    batch_attn_size_sent = array_ops.stack([batch_size, attn_size_sent])

    attns_doc = [
        array_ops.zeros(
            batch_attn_size_doc, dtype=dtype) for _ in xrange(num_heads)
    ]

    attns_sent = [
      array_ops.zeros(
        batch_attn_size_sent, dtype=dtype) for _ in xrange(num_heads)
    ]

    for a_doc in attns_doc:  # Ensure the second shape of attention vectors is set.
      a_doc.set_shape([None, attn_size_doc])

    for a_sent in attns_sent:  # Ensure the second shape of attention vectors is set.
      a_sent.set_shape([None, attn_size_sent])

    if initial_state_attention:
      attns_doc, attn_doc_value = attention_doc(initial_state)
      attns_sent, _ = attention_sent(initial_state, attn_doc_value)

    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + attns_doc + attns_sent, input_size, True)
      #x = linear([inp], input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns_doc, attn_doc_value = attention_doc(state)
          attns_sent, _ = attention_sent(state, attn_doc_value)
      else:
        attns_doc, attn_doc_value = attention_doc(state)
        attention2encoder_doc.append(attn_doc_value)
        attns_sent, attn_sent_value = attention_sent(state, attn_doc_value)
        attention2encoder_sent.append(attn_sent_value)
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns_doc + attns_sent, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  # outputs: dec_timesteps*[batch_size, cell.output_size]
  # state: ([batch_size, cell.c_state_size], [batch_size, cell.h_state_size])
  # attention: [batch_size, attn_length]
  return outputs, state, math_ops.reduce_mean(array_ops.stack(attention2encoder_doc), axis=0), \
         math_ops.reduce_mean(array_ops.stack(attention2encoder_sent), axis=0)

def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(target, logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps

def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost

def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, core_rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputs, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses
