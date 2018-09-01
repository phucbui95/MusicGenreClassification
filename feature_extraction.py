# coding: utf-8

# In[6]:


from IPython.display import Audio
import librosa
from librosa import display
import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import os

from multiprocessing import Process


col_names = ['file_name', 'frame', 'signal_mean', 'signal_std', 'signal_skew', 'signal_kurtosis',
             'zcr_mean', 'zcr_std', 'rmse_mean', 'rmse_std', 'tempo',
             'spectral_centroid_mean', 'spectral_centroid_std',
             'spectral_bandwidth_2_mean', 'spectral_bandwidth_2_std',
             'spectral_bandwidth_3_mean', 'spectral_bandwidth_3_std',
             'spectral_bandwidth_4_mean', 'spectral_bandwidth_4_std'] + \
            ['spectral_contrast_' + str(i + 1) + '_mean' for i in range(5)] + \
            ['spectral_contrast_' + str(i + 1) + '_std' for i in range(5)] + \
            ['spectral_rolloff_mean', 'spectral_rolloff_std'] + \
            ['mfccs_' + str(i + 1) + '_mean' for i in range(20)] + \
            ['mfccs_' + str(i + 1) + '_std' for i in range(20)] + \
            ['chroma_stft_' + str(i + 1) + '_mean' for i in range(12)] + \
            ['chroma_stft_' + str(i + 1) + '_std' for i in range(12)]

# In[167]:

def extract_features(filepath, sr=22050):
    # Read wav-file
    file_id = filepath.split("/")[-1].split(".")[0]

    if filepath.split(".")[-1] == "mp3":
        s, sr = librosa.load(filepath, sr=sr)

        print(len(s), sr)
        frame_len = len(s) // 4
        x = np.zeros((4, frame_len))
        for i in range(4):
            x[i, :] = s[frame_len * i: frame_len*i + frame_len]
    else:
        x, sr = np.load(os.path.join("data", "raw", "{}.npy".format(file_id))), 12000

    seg_features = []
    for i in range(x.shape[0]):
        y = x[i, :].reshape(-1)
        feature_list = [file_id]

        feature_list.append(np.mean(abs(y)))
        feature_list.append(np.std(y))
        feature_list.append(scipy.stats.skew(abs(y)))
        feature_list.append(scipy.stats.kurtosis(y))

        zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
        feature_list.append(np.mean(zcr))
        feature_list.append(np.std(zcr))

        rmse = librosa.feature.rmse(y + 0.0001)[0]
        feature_list.append(np.mean(rmse))
        feature_list.append(np.std(rmse))

        tempo = librosa.beat.tempo(y, sr=sr)
        feature_list.extend(tempo)

        spectral_centroids = librosa.feature.spectral_centroid(y + 0.01, sr=sr)[0]
        feature_list.append(np.mean(spectral_centroids))
        feature_list.append(np.std(spectral_centroids))

        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=2)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=4)[0]
        feature_list.append(np.mean(spectral_bandwidth_2))
        feature_list.append(np.std(spectral_bandwidth_2))
        feature_list.append(np.mean(spectral_bandwidth_3))
        feature_list.append(np.std(spectral_bandwidth_3))
        feature_list.append(np.mean(spectral_bandwidth_3))
        feature_list.append(np.std(spectral_bandwidth_3))

        spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands=4, fmin=300.0)
        feature_list.extend(np.mean(spectral_contrast, axis=1))
        feature_list.extend(np.std(spectral_contrast, axis=1))

        spectral_rolloff = librosa.feature.spectral_rolloff(y + 0.01, sr=sr, roll_percent=0.85)[0]
        feature_list.append(np.mean(spectral_rolloff))
        feature_list.append(np.std(spectral_rolloff))

        mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
        feature_list.extend(np.mean(mfccs, axis=1))
        feature_list.extend(np.std(mfccs, axis=1))

        chroma_stft = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)
        feature_list.extend(np.mean(chroma_stft, axis=1))
        feature_list.extend(np.std(chroma_stft, axis=1))

        feature_list[1:] = np.round(feature_list[1:], decimals=3)

        seg_features.append(feature_list)

    return sum(seg_features, [])


def extract_short_term_features(filepath, sr=22050):
    # Read wav-file
    file_id = filepath.split("/")[-1].split(".")[0]
    s, sr = librosa.load(filepath, sr=sr)
        # frame_len = len(s) // 4
        # x = np.zeros((4, frame_len))
        # for i in range(4):
        #     x[i, :] = s[frame_len * i: frame_len*i + frame_len]
    dura = 5
    frame_size = dura * sr


    seg_features = []
    for i in range(0, len(s) - (len(s) % frame_size), frame_size):
        y = s[i : i + frame_size]
        feature_list = [file_id, i / frame_size]

        feature_list.append(np.mean(abs(y)))
        feature_list.append(np.std(y))
        feature_list.append(scipy.stats.skew(abs(y)))
        feature_list.append(scipy.stats.kurtosis(y))

        zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
        feature_list.append(np.mean(zcr))
        feature_list.append(np.std(zcr))

        rmse = librosa.feature.rmse(y + 0.0001)[0]
        feature_list.append(np.mean(rmse))
        feature_list.append(np.std(rmse))

        tempo = librosa.beat.tempo(y, sr=sr)
        feature_list.extend(tempo)

        spectral_centroids = librosa.feature.spectral_centroid(y + 0.01, sr=sr)[0]
        feature_list.append(np.mean(spectral_centroids))
        feature_list.append(np.std(spectral_centroids))

        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=2)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=4)[0]
        feature_list.append(np.mean(spectral_bandwidth_2))
        feature_list.append(np.std(spectral_bandwidth_2))
        feature_list.append(np.mean(spectral_bandwidth_3))
        feature_list.append(np.std(spectral_bandwidth_3))
        feature_list.append(np.mean(spectral_bandwidth_3))
        feature_list.append(np.std(spectral_bandwidth_3))

        spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands=4, fmin=300.0)
        feature_list.extend(np.mean(spectral_contrast, axis=1))
        feature_list.extend(np.std(spectral_contrast, axis=1))

        spectral_rolloff = librosa.feature.spectral_rolloff(y + 0.01, sr=sr, roll_percent=0.85)[0]
        feature_list.append(np.mean(spectral_rolloff))
        feature_list.append(np.std(spectral_rolloff))

        mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
        feature_list.extend(np.mean(mfccs, axis=1))
        feature_list.extend(np.std(mfccs, axis=1))

        chroma_stft = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)
        feature_list.extend(np.mean(chroma_stft, axis=1))
        feature_list.extend(np.std(chroma_stft, axis=1))

        feature_list[1:] = np.round(feature_list[1:], decimals=3)

        seg_features.append(feature_list)

    return seg_features

def parallel_compute_feature(df_data, data_folder):
    #extract_feature(df_data, test_data_folder, "feature_test.csv")
    args = []
    nthreads = 8
    num_batch = len(df_data) // nthreads
    # os.makedirs("data/features")
    for i in range(nthreads):
        output = os.path.join("data", "features")
        if i == nthreads - 1 :
            args.append((df_data[i * num_batch : ], output))
        else:
            args.append((df_data[i * num_batch : i * num_batch + num_batch], output))

    processes = [Process(target=extract_feature, args=(args[i][0], data_folder, args[i][1],)) for i in range(nthreads)]

    for p in processes: p.start()
    for p in processes: p.join()

def appendDFToCSV_void(df, csvFilePath, sep=","):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

def extract_feature(df_data, data_folder, output="feature.csv"):
    # columns = ["{}@{}".format(c, x) for x in range(4) for c in col_names]

    for i in range(len(df_data)):
        if not os.path.exists(output):
            os.makedirs(output)
        output_fullpath = os.path.join(output, "feature_{}.csv".format(df_data.iloc[i, 0].split(".")[0]))
        if os.path.exists(output_fullpath):
            continue
        # try:
        ft = extract_short_term_features(os.path.join(data_folder, df_data.iloc[i, 0]))
        frames = []
        for s_ft in ft:
            df_frame = pd.DataFrame([s_ft], columns=col_names)
            frames.append(df_frame)
        df_frames = pd.concat(frames)

        df_frames.to_csv(output_fullpath, index=None)
        if i % 200 == 0:
            print("Proc: {:.3f}".format(float(i) / len(df_data)))
        # except:

if __name__ == '__main__':
    # Load train data
    df_train = pd.read_csv(os.path.join("data", "train.csv"), header=None)
    df_test = pd.read_csv(os.path.join("data", "test.csv"), header=None)
    df_train.columns = ["file", "genre_code"]
    df_genre = pd.read_csv(os.path.join("data", "genres.csv"))

    train_data_folder = os.path.join("data", "mp3", "train")
    test_data_folder = os.path.join("data", "mp3", "test")

    parallel_compute_feature(df_train[:10], train_data_folder)
