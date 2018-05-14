import os
from os.path import join

import librosa
import numpy as np
import soundfile as sf
import pyworld as pw
import tensorflow as tf
from config import *
import random

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dir_to_wav', './dataset/vcc2016/wav', 'Dir to *.wav')
tf.app.flags.DEFINE_string('dir_to_bin', './dataset/vcc2016/bin', 'Dir to output *.bin')
tf.app.flags.DEFINE_integer('fs', 16000, 'Global sampling frequency')
tf.app.flags.DEFINE_float('f0_ceil', 500, 'Global f0 ceiling')

SPEAKERS = [s.strip() for s in tf.gfile.GFile('./etc/speakers.tsv', 'r').readlines()]

def read_features_from_file(f):
    features = np.fromfile(f, np.float32)
    features = np.reshape(features, [-1, 513*2 + 1 + 1 + 1]) # f0, en, spk
    return {
        'sp': features[:, :SP_DIM],
        'ap': features[:, SP_DIM : 2*SP_DIM],
        'f0': features[:, SP_DIM * 2],
        'en': features[:, SP_DIM * 2 + 1],
        'label': features[:, SP_DIM * 2 + 2],
        'feature': features[:, :SP_DIM * 2 + 1],
        'filename': f,
    }

def pw2wav(features, feat_dim=513, fs=16000):
    ''' NOTE: Use `order='C'` to ensure Cython compatibility '''
    en = np.reshape(features['en'], [-1, 1])
    sp = np.power(10., features['sp'])
    sp = en * sp
    if isinstance(features, dict):
        return pw.synthesize(
            features['f0'].astype(np.float64).copy(order='C'),
            sp.astype(np.float64).copy(order='C'),
            features['ap'].astype(np.float64).copy(order='C'),
            fs,
        )
    features = features.astype(np.float64)
    sp = features[:, :feat_dim]
    ap = features[:, feat_dim:feat_dim*2]
    f0 = features[:, feat_dim*2]
    en = features[:, feat_dim*2 + 1]
    en = np.reshape(en, [-1, 1])
    sp = np.power(10., sp)
    sp = en * sp
    return pw.synthesize(
        f0.copy(order='C'),
        sp.copy(order='C'),
        ap.copy(order='C'),
        fs
    )

def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    ''' Extract WORLD feature from waveform '''
    _f0, t = pw.dio(x, fs, f0_ceil=args.f0_ceil)            # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size) # extract aperiodicity
    return {
        'f0': f0,
        'sp': sp,
        'ap': ap,
    }


def extract(filename, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction '''
    x, _ = librosa.load(filename, sr=args.fs, mono=True, dtype=np.float64)
    features = wav2pw(x, args.fs, fft_size=fft_size)
    ap = features['ap']
    f0 = features['f0'].reshape([-1, 1])
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    sp = np.log10(sp / en)
    return np.concatenate([sp, ap, f0, en], axis=1).astype(dtype)


def extract_and_save_bin_to(dir_to_bin, dir_to_source):
    sets = [s for s in os.listdir(dir_to_source) if s in SETS]
    for d in sets:
        path = join(dir_to_source, d)
        speakers = [s for s in os.listdir(path) if s in SPEAKERS]
        for s in speakers:
            path = join(dir_to_source, d, s)
            output_dir = join(dir_to_bin, d, s)
            if not tf.gfile.Exists(output_dir):
                tf.gfile.MakeDirs(output_dir)
            for f in os.listdir(path):
                filename = join(path, f)
                print(filename)
                if not os.path.isdir(filename):
                    features = extract(filename)
                    labels = SPEAKERS.index(s) * np.ones(
                        [features.shape[0], 1],
                        np.float32,
                    )
                    b = os.path.splitext(f)[0]
                    features = np.concatenate([features, labels], 1)
                    with open(join(output_dir, '{}.bin'.format(b)), 'wb') as fp:
                        fp.write(features.tostring())


def get_files_and_que(file_pattern):
    files = tf.gfile.Glob(file_pattern)
    random.shuffle(files)
    filename_queue = tf.train.string_input_producer(files)

    return files, filename_queue

def get_batch(files, offset, batch_size):
    batch_files = [files[i % len(files)] for i in range(offset * batch_size, (offset+1) * batch_size)]
    batch_x = []
    batch_y = []
    for file in batch_files:
        features = read_features_from_file(file)

        sp = features['sp']
        sp = np.pad(sp, ((0,2500-sp.shape[0]),(0,0)), "constant", constant_values=0)
        label = np.zeros(10)
        np.put(label,int(features['label'][0]),1)

        batch_x.append(sp)
        batch_y.append(label)

    return np.asarray(batch_x), np.asarray(batch_y)


def main():
    file_pattern = "./dataset/vcc2016/bin/Training Set/SF3/1000*.bin"
    files, filename_queque = get_files_and_que(file_pattern)

    for file in files:
        features = read_features_from_file(file)
        print(features['label'][0])



    # extract_and_save_bin_to(
    #     args.dir_to_bin,
    #     args.dir_to_wav,
    # )

    # f = './dataset/vcc2016/bin/Training Set/SF1/100001.bin'
    # features = read_features_from_file(f)
    #
    # y = pw2wav(features)
    # sf.write('test2.wav', y, 16000)

    # file_pattern = "./dataset/vcc2016/bin/Training Set/*/1000*.bin"
    # batch_size = 16
    # files, filename_queue = get_files_and_que(file_pattern)
    #
    # for i in range(0, len(files) / batch_size + 1):
    #     batch_x, batch_y = get_batch(files, i, batch_size)
    #     print(batch_y)


if __name__ == '__main__':
    main()
