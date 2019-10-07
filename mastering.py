
import util
import numpy as np
import librosa
import glob
import os
from tqdm import tqdm

# TODO: https://bakuage.com/blog/youtube-loudness-normalization/ に元ずいた変換を試したい

BASE_WAVES_PATH = './datasets/suzuki'
CONVERT_WAVES_PATH = './datasets/kinoshita'
DIST_WAVES_PATH = './datasets/kinoshita/overdrive'
SAMPLE_RATE = 44100

def main():
    m = getMax(BASE_WAVES_PATH)
    convert(CONVERT_WAVES_PATH, DIST_WAVES_PATH, m)
    print(m)

def getMax(dir: str, sr: int = SAMPLE_RATE) -> float:
    print('find max...')
    all_max = 0
    for f in tqdm(glob.glob(os.path.join(dir, '*.wav'))):
        wave, _ = librosa.load(f, sr=sr, mono=True)
        maxv = wave.max()
        maxv = abs(maxv)
        minv = wave.min()
        minv = abs(minv)
        all_max = max([all_max, maxv, minv])
    return all_max


def convert(dir_src: str, dir_dist: str, max_value: float, sr: int = SAMPLE_RATE):
    files = []
    waves = []
    A_max = max_value
    B_max = 0
    print('find max...')
    for f in tqdm(glob.glob(os.path.join(dir_src, '*.wav'))):
        files.append(os.path.basename(f))
        wave, _ = librosa.load(f, sr=sr, mono=True)
        waves.append(wave)
        maxv = wave.max()
        maxv = abs(maxv)
        minv = wave.min()
        minv = abs(minv)
        B_max = max([B_max, maxv, minv])
    print('convert wav file...')
    for wave, f in tqdm(zip(waves, files)):
        dist = wave*(A_max/B_max)
        librosa.output.write_wav(os.path.join(dir_dist, f), dist, sr)

if __name__ == "__main__":
    main()
