from vggish_preprocess import preprocess_sound
import numpy as np
from numpy.random import seed, randint
from scipy.io import wavfile
from sklearn import svm
import os
from tqdm import tqdm

def properties_of_dataset(file):
    files = os.listdir(file)
    wav_length = []
    sr_list = []
    for class_val, directory in enumerate(files):
        path = file+directory+'/'
        paths = os.listdir(path)
        for music in tqdm(paths, desc='Class '+directory, total=len(paths)): #path+music
            try:
                sr, wav_data = wavfile.read(path+music)
            except:
                continue
            sr_list.append(sr)
            wav_length.append(wav_data.shape[0])
    return sr_list, wav_length

def loading_wav(file, steps, seg_len=5):
    data = []
    sound_file = file
    sr, wav_data = wavfile.read(sound_file)
    length = sr * seg_len           # 5s segment
    if wav_data.shape[0]//length > 0:
        for j in range(0, wav_data.shape[0] - length, steps):
            cur_wav = wav_data[j:(j+1)+length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data.append(cur_spectro)
    else:
        cur_wav = np.concatenate((wav_data, np.zeros(length - wav_data.shape[0])))
        cur_wav = cur_wav / 32768.0
        cur_spectro = preprocess_sound(cur_wav, sr)
        cur_spectro = np.expand_dims(cur_spectro, 3)
        data.append(cur_spectro)
    return np.array(data)

def loading_dataset(file, steps, seg_len=5):
    files = os.listdir(file)
    dataset_x = []
    dataset_y = []
    label_map = {}
    first = True
    for class_val, directory in enumerate(files):
        label_map.update({class_val: directory})
        path = file+directory+'/'
        paths = os.listdir(path)
        for music in tqdm(paths, desc='Class '+directory, total=len(paths)):
            try:
                x_dash = loading_wav(path+music, steps, seg_len)
            except:
                continue
            dataset_x.append(x_dash)
            dataset_y.append(np.full((x_dash.shape[0], 1), class_val))

    dataset_x, dataset_y = np.array(dataset_x), np.array(dataset_y)

    x, y = dataset_x, dataset_y

    dataset_x = []
    dataset_y = []

    for x_val, y_val in zip(x, y):
        for i in range(x_val.shape[0]):
            dataset_x.append(x_val[i][0])
            dataset_y.append(y_val[i][0])

    dataset_x, dataset_y = np.array(dataset_x), np.array(dataset_y)

    return dataset_x, dataset_y, label_map