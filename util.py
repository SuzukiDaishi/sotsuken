import librosa
import numpy as np
import pyworld
import glob
import pickle
from tqdm import tqdm
import os
from typing import Tuple, List

SAMPLE_RATE = 44100

def loadWave(file: str, sr: int = SAMPLE_RATE) -> np.ndarray :
    '''
    音声ファイルを読み込む

    Parameters
    ----------
    file: str
        ファイルのパス
    sr: int, default SAMPLE_RATE
        サンプルレート
    
    Returns
    -------
    wave: np.ndarray
        音声の波形
    '''
    wave, _ = librosa.load(file, sr=sr, mono=True)
    return wave

def wavePadding(wave: np.ndarray, sr: int = SAMPLE_RATE, frame_period: float = 5., multiple: int = 4) -> np.ndarray :
    '''
    音声の前と後ろを0で埋める

    Parameters
    ----------
    wave: np.ndarray
        波形データ
    sr: int, default SAMPLE_RATE
        サンプルレート
    frame_period: float, default 5.
        フレームの間隔[msec]
    multiple: int, default 4
        TODO: わからんから調べる
    
    Returns
    -------
    wave_padded: np.ndarray
        パディングされた波形
    '''
    assert wave.ndim == 1
    num_frames = len(wave)
    num_frames_padded = int((np.ceil((np.floor(num_frames / 
                        (sr * frame_period / 1000)) + 1) / 
                        multiple + 1) * multiple - 1) 
                        * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wave_padded = np.pad(wave, (num_pad_left, num_pad_right),
                  'constant', constant_values=0)
    return wave_padded


def worldDecompose(wave: np.ndarray, fs: int = SAMPLE_RATE, 
                   frame_period: float = 5.) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    音声をworldを用いてf0, spectral envelope, aperiodicityに分解

    Parameters
    ----------
    wave: np.ndarray
        音声の波形データ
    fs: int, default SAMPLE_RATE
        サンプリング周波数
    frame_period: float, default 5.
        フレームの時間的間隔

    Returns
    -------
    f0: np.ndarray
        フレームの基本周波数[hz]
    sp: np.ndarray
        スペクトル包絡
    ap: np.ndarray
        非周期性指標
    '''
    wave = wave.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wave, fs, frame_period=frame_period, f0_floor=71., f0_ceil=800.)
    sp = pyworld.cheaptrick(wave, f0, timeaxis, fs)
    ap = pyworld.d4c(wave, f0, timeaxis, fs)
    return f0, sp, ap

def pitchConversion(f0: np.ndarray, mean_log_src: float, std_log_src: float, 
                    mean_log_target: float, std_log_target: float) -> np.ndarray:
    '''
    対数正規分布を用いてF0のピッチを変更する

    Parameters
    ----------
    f0: np.ndarray
        フレームの基本周波数[hz]
    mean_log_src: float
        変更前のドメインのF0の平均のLog
    std_log_src: float
        変更前のドメインのF0の標準偏差のLog
    mean_log_target: float
        変更後のドメインのF0の平均のLog
    std_log_target: float
        変更後のドメインのF0の標準偏差のLog
    
    Returns
    -------
    f0_converted: np.ndarray
        ピッチ変更後のF0
    '''
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)
    return f0_converted


def worldEncodeSpectralEnvelop(sp: np.ndarray, fs: int = SAMPLE_RATE, dim: int = 36) -> np.ndarray :
    '''
    スペクトル包絡を元にMCEPsをつくる

    Parameters
    ----------
    sp: np.ndarray
        スペクトル包絡のデータ
    fs: int, default SAMPLE_RATE
        サンプリング周波数
    dim: int, default 24
        iFFTの次元数
    
    Returns
    -------
    code_spectral_envelope: np.ndarray
        スペクトル包絡のMCEPs
    '''
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp


def worldDecodeSpectralEnvelop(coded_sp: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray :
    '''
    MCEPsをスペクトル包絡に戻す

    Parameters
    ----------
    coded_sp: np.ndarray
        MCEPsのデータ
    fs: int, default SAMPLE_RATE
        サンプリング周波数
    
    Returns
    -------
    decoded_sp: np.ndarray
        スペクトル包絡
    '''
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp


def worldSpeechSynthesis(f0: np.ndarray, decoded_sp: np.ndarray, ap: np.ndarray, 
                         fs: int = SAMPLE_RATE, frame_period: float = 5.) -> np.ndarray :
    '''
    worldでシンセサイズする

    Parameters
    ----------
    f0: np.ndarray
        フレームの基本周波数[hz]
    decoded_sp: np.ndarray
        スペクトル包絡
    ap: np.ndarray
        非周期性指標
    fs: int, default SAMPLE_RATE
        サンプリング周波数
    frame_period: float, default 5.
        フレームの時間的間隔
    
    Returns
    -------
    wave: np.ndarray
        合成した波形データ
    '''
    wave = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    wave = wave.astype(np.float32)
    return wave

def savePickle(path: str, obj: any) -> None :
    '''
    pickleで保存

    Parameters
    ----------
    path: str
        保存先のパス
    obj: any
        保存するデータ
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def loadPickle(path: str) -> any :
    '''
    pickleをロード

    Parameters
    ----------
    path: str
        ロードするファイル
    
    Returns
    -------
    obj: any
        ロードするデータ
    '''
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj

def loadWaves(dir: str, sr: int = SAMPLE_RATE) -> List[np.ndarray] :
    '''
    音声ファイルを一括でロードする

    Parameters
    ----------
    dir: str
        wavファイルのあるディレクトリ
    sr: int, default SAMPLE_RATE
        サンプルレート

    Returns
    -------
    waves: List[np.ndarray]
        音声データの入った配列
    '''
    print('Loading Wavs...')
    waves = []
    for f in tqdm(glob.glob(os.path.join(dir, '*.wav'))):
        wave, _ = librosa.load(f, sr=sr, mono=True)
        waves.append(wave)
    return waves


def worldEncodeData(waves: List[np.ndarray], fs: int = SAMPLE_RATE, frame_period: float = 5., 
                    coded_dim: int = 36) -> Tuple[List[np.ndarray], List[np.ndarray], 
                    List[np.ndarray], List[np.ndarray]]:
    '''
    worldを用いて音声から特徴量を抽出する

    Parameters
    ----------
    waves: List[np.ndarray]
        音声データの入った配列
    fs: int, default SAMPLE_RATE
        サンプルレート
    frame_period: float, default 5.
        フレームの間隔[msec]
    coded_dim: int, default 36
        iFFTの際の次元数
    
    Retruns 
    -------
    f0s: List[np.ndarray]
        フレームの基本周波数[hz]の配列
    sps: List[np.ndarray]
        スペクトル包絡の配列
    aps: List[np.ndarray]
        非周期性指標の配列
    coded_sps: List[np.ndarray]
        スペクトル包絡のMCEPs
    '''
    print('Extracting acoustic features...')
    f0s, sps, aps, coded_sps = [], [], [], []
    for wave in tqdm(waves):
        f0, sp, ap = worldDecompose(wave, fs=fs, frame_period=frame_period)
        coded_sp = worldEncodeSpectralEnvelop(sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)
    return f0s, sps, aps, coded_sps


def logf0Statistics(f0s: List[np.ndarray]) -> Tuple[float, float] :
    '''
    F0のlogの平均,標準偏差を求める

    Parameters
    ----------
    f0s: List[np.ndarray]
        フレームの基本周波数[hz]の配列
    
    Returns
    -------
    log_f0s_mean: float
        FOのLogの平均
    log_f0s_std: float
        F0のLogの標準偏差
    '''
    print('Calculating F0 statistics...')
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()
    print('Mean: %f, Std: %f' % (log_f0s_mean, log_f0s_std))
    return log_f0s_mean, log_f0s_std


def transposeInList(lst: List[np.ndarray]) -> List[np.ndarray] :
    '''
    配列の中のnumpyを全て転置する

    Parameters
    ----------
    lst: List[np.ndarray]
        numpyの配列
    
    Returns
    -------
    transposed_lst: List[np.ndarray]
        転置したnumpyの配列
    '''
    transposed_lst = []
    for arr in lst:
        transposed_lst.append(arr.T)
    return transposed_lst

def codedSpsNormalizationFitTransoform(coded_sps: List[np.ndarray], use_max: bool = False) -> Tuple[List[np.ndarray], List[float], 
                                       List[float], List[float]]:
    '''
    MCEPsの配列を正規化

    Parameters
    ----------
    coded_sps: List[np.ndarray] 
        MCEPsの配列
    use_max: bool
        元のコードではstdで正規化しているが
        気持ち悪い場合はTrueにしてください

    Returns
    -------
    coded_sps_normalized: List[np.ndarray]
        正規化したMCEPsの配列
    coded_sps_mean: List[float]
        正規化したMCEPsの平均
    coded_sps_std: List[float]
        正規化したMCEPsの標準偏差
    coded_sps_max: List[float]
        正規化したMCEPsの最大値
    
    Note
    ----
    絶対なんかおかしい
    1. maxのshapeがベクトル
    2. maxは負値を考慮できていない
    3. maxはmeanを引いた値をベースにしなければならない
    4. なのに変えたら動かない(シンセサイズ時にnanになる)
    '''
    print('Normalizing data...')
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_max = np.max(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_normalized = []
    for coded_sp in tqdm(coded_sps):
        if use_max:
            coded_sp_norm = (coded_sp - coded_sps_mean) / coded_sps_max
        else:
            coded_sp_norm = (coded_sp - coded_sps_mean) / coded_sps_std
        coded_sps_normalized.append(coded_sp_norm)
    return coded_sps_normalized, coded_sps_mean, coded_sps_std, coded_sps_max


def codedSpsNormalizationFitTransoform_fix(coded_sps: List[np.ndarray], use_max: bool = False) -> Tuple[List[np.ndarray], List[float],
                                                                                                    List[float], List[float]]:
    '''
    MCEPsの配列を正規化

    Parameters
    ----------
    coded_sps: List[np.ndarray] 
        MCEPsの配列
    use_max: bool
        元のコードではstdで正規化しているが
        気持ち悪い場合はTrueにしてください

    Returns
    -------
    coded_sps_normalized: List[np.ndarray]
        正規化したMCEPsの配列
    coded_sps_mean: List[float]
        正規化したMCEPsの平均
    coded_sps_std: List[float]
        正規化したMCEPsの標準偏差
    coded_sps_max: List[float]
        正規化したMCEPsの最大値

    '''
    print('Normalizing data...')
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    _max = np.abs(np.max(coded_sps_concatenated - coded_sps_mean, axis=1, keepdims=True))
    _min = np.abs(np.min(coded_sps_concatenated - coded_sps_mean, axis=1, keepdims=True))
    coded_sps_max = np.concatenate([_max, _min], axis=1).max(axis=1, keepdims=True)
    coded_sps_normalized = []
    for coded_sp in tqdm(coded_sps):
        coded_sp_norm = (coded_sp - coded_sps_mean) / coded_sps_max
        coded_sps_normalized.append(coded_sp_norm)
    return coded_sps_normalized, coded_sps_mean, coded_sps_std, coded_sps_max
