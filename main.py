from audio_preprocessing import *

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from urllib import urlretrieve
from six.moves import cPickle as pickle
import pandas as pd
from music_tagger_crnn import MusicTaggerCRNN

import librosa
import librosa.display

# Download metadata
url = "https://dl.challenge.zalo.ai/music/"
ROOT_PATH = "data"
MUSIC_PATH = os.path.join("..", "music_genre", "train")

def maybe_download(filename, expected_bytes=None, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, os.path.join(ROOT_PATH, filename))
  statinfo = os.stat(filename)
  if expected_bytes is None or statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return filename

if __name__ == '__main__':

  # maybe_download("train.csv")
  # maybe_download("test.csv")
  # maybe_download("genres.csv")

  # Load train data
  df_train = pd.read_csv(os.path.join("data", "train.csv"), header=None)
  df_train.columns = ["file", "genre_code"]
  df_genre = pd.read_csv(os.path.join("data", "genres.csv"))

  test_mp3  = df_train.iloc[2][0]
  #melgram = compute_melgram(audio_path=os.path.join(MUSIC_PATH, test_mp3))
  #print(melgram.shape)
  train_folder = os.path.join("data", "train")
  if not os.path.exists(train_folder):
    os.makedirs(train_folder)

  for i in range(len(df_train)):
    filename = df_train.iloc[i][0]
    file_id = filename.split(".")[0]
    output_path = os.path.join(train_folder, file_id)
    if os.path.exists(os.path.join(train_folder, "{}.npy".format(file_id))):
      continue
    load_audio(os.path.join(MUSIC_PATH, filename), output_path)
    if i % 100 == 0:
      print("Processed %0.2f" % (100 * i / len(df_train)))

  print("Done")