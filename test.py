import pysptk
import pyworld
import config
import librosa
import numpy as np


def get_features(wav_path):
    x, fs = librosa.load(wav_path, sr=config.fs)
    x = x.astype(np.float64)
    f0, time_axis = pyworld.dio(x, fs, frame_period=config.frame_period)
    f0 = pyworld.stonemask(x, f0, time_axis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, time_axis, fs)
    aperiodicity = pyworld.d4c(x, f0, time_axis, fs)
    mc = pysptk.sp2mc(spectrogram, order=config.order, alpha=config.alpha)
    return mc, aperiodicity, f0


def main():
    wav_path = "/DATA2/data/qhhuang/vcc/vcc2018_training/VCC2SF1/10001.wav"
#    wav_path = "/Users/mc/Documents/Study/zhangya's lab/data/vcc2016+2018/vcc2018_training/VCC2SF1/10001.wav"
    mc, aperiodicity, f0 = get_features(wav_path)

    print(mc.shape)
    print(f0.shape)
    print(aperiodicity.shape)

    spectrogram = pysptk.mc2sp(
        mc.astype(np.float64), alpha=config.alpha, fftlen=config.fftlen)
    waveform = pyworld.synthesize(
        f0, spectrogram, aperiodicity, config.fs, config.frame_period)

    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(
        'out.wav', (waveform * maxv).astype(np.int16), config.fs)


if __name__ == '__main__':
    main()
