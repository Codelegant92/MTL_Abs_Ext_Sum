"""Data batchers for data described in ..//data_prep/README.md."""

import glob
import random
import struct
import sys
import numpy as np

from tensorflow.core.example import example_pb2

# Special tokens
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

class Vocab(object):
  """Vocabulary class for mapping words and ids."""

  def __init__(self, vocab_file, emb_file, emb_dim, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0
    self._word = []

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          sys.stderr.write('Bad line: %s\n' % line)
          continue
        if pieces[0] in self._word_to_id:
          raise ValueError('Duplicated word: %s.' % pieces[0])
        self._word_to_id[pieces[0]] = self._count
        self._id_to_word[self._count] = pieces[0]
        self._word.append(pieces[0])
        self._count += 1
        if self._count > max_size:
          break
          #raise ValueError('Too many words: >%d.' % max_size)
    '''
    self._vocab_embedding = np.zeros([self._count, emb_dim], dtype=np.float32)
    num = 0
    with open(emb_file, 'r') as emb_f:
      while num < max_size * 5:
        row = emb_f.readline()
        word = row.strip().split(' ')[0]
        if word in self._word:
          word_idx = self._word.index(word)

          vector_str = row.strip().split(' ')[1:]
          vector = np.array([float(item) for item in vector_str], dtype=np.float32)
          self._vocab_embedding[word_idx][:] = vector
          if num % 10000 == 0:
            print "%d ====> word: %s; idx: %d" % (num, word, word_idx)
        num += 1

    emb_f.close()

    for i in range(self._count):
      if (not np.any(self._vocab_embedding[i][:])):
        self._vocab_embedding[i][:] = np.random.normal(0.0, 0.5, emb_dim)
    '''
  def CheckVocab(self, word):
    if word not in self._word_to_id:
      return None
    return self._word_to_id[word]
  
  def WordToId(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def IdToWord(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('id not found in vocab: %d.' % word_id)
    return self._id_to_word[word_id]

  def NumIds(self):
    return self._count
  '''
  def VocabEmbedding(self):
    return self._vocab_embedding
  '''
def ExampleGen(data_path, num_epochs=None):
  """Generate articles or abstracts or labels from data files"""
  epoch = 0
  while True:
    if num_epochs is not None and epoch >= num_epochs:
      break
    with open(data_path, 'r') as f:
      while True:
        line = f.readline().strip()
        if line:
          yield line
        else:
          break
    epoch += 1

def Pad(ids, pad_id, length):
  """Pad or trim list to len length.

  Args:
    ids: list of ints to pad
    pad_id: what to pad with
    length: length to pad or trim to

  Returns:
    ids trimmed or padded with pad_id
  """
  assert pad_id is not None
  assert length is not None

  if len(ids) < length:
    a = [pad_id] * (length - len(ids))
    return ids + a
  else:
    return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
  """Get ids corresponding to words in text.

  Assumes tokens separated by space.

  Args:
    text: a string
    vocab: TextVocabularyFile object
    pad_len: int, length to pad to
    pad_id: int, word id for pad symbol

  Returns:
    A list of ints representing word ids.
  """
  ids = []
  for w in text.split():
    i = vocab.WordToId(w)
    if i >= 0:
      ids.append(i)
    else:
      ids.append(vocab.WordToId(UNKNOWN_TOKEN))
  if pad_len is not None:
    return Pad(ids, pad_id, pad_len)
  return ids


def Ids2Words(ids_list, vocab):
  """Get words from ids.

  Args:
    ids_list: list of int32
    vocab: TextVocabulary object

  Returns:
    List of words corresponding to ids.
  """
  assert isinstance(ids_list, list), '%s  is not a list' % ids_list
  return [vocab.IdToWord(i) for i in ids_list]


def SnippetGen(text, start_tok, end_tok, inclusive=True):
  """Generates consecutive snippets between start and end tokens.

  Args:
    text: a string
    start_tok: a string denoting the start of snippets
    end_tok: a string denoting the end of snippets
    inclusive: Whether include the tokens in the returned snippets.

  Yields:
    String snippets
  """
  cur = 0
  while True:
    try:
      start_p = text.index(start_tok, cur)
      end_p = text.index(end_tok, start_p + 1)
      cur = end_p + len(end_tok)
      if inclusive:
        yield text[start_p:cur]
      else:
        yield text[start_p+len(start_tok):end_p]
    except ValueError as e:
      raise StopIteration('no more snippets in text: %s' % e)

def ToSentences(paragraph, include_token=True):
  """Takes tokens of a paragraph and returns list of sentences.

  Args:
    paragraph: string, text of paragraph
    include_token: Whether include the sentence separation tokens result.

  Returns:
    List of sentence strings.
  """
  s_gen = SnippetGen(paragraph, SENTENCE_START, SENTENCE_END, include_token)
  return [s for s in s_gen]
