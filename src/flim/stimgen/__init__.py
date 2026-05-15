"""
エージェントを訓練するために使用する実験を模したデータを生成する
"""
from collections.abc import Sequence
from typing import Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pywavelet as w
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft
from torch.utils.data import DataLoader, Dataset

T = TypeVar("T")


class FLKLDataset(Dataset[T], Generic[T]):
    def __init__(self, items: Sequence[T]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

###########################
# Visual flickr generator #
###########################

_N = 11
_TAU = 10
_T = int(2000 / _TAU)
T = _T
_PULSE_DURATION = int(10 / _TAU)

def generate_pulse_onsets(freq: float, t: int = _T, shift: int=0):
    interval = t / (freq * (_TAU * t / 1000))
    expct_npulse = int((_T - 1) / interval)
    onsets = np.cumsum(np.full(expct_npulse + 10, interval))
    shift += 1
    noise = np.random.uniform(-shift, shift, expct_npulse + 10).astype(np.int64)
    onsets = onsets + noise
    # onsets = np.cumsum(np.full(expct_npulse + 10, interval) + np.random.uniform(-shift, shift, expct_npulse + 10)).astype(np.int64)
    # return np.append(0, onsets[onsets < _T]) # こっちでやるとノイズが+方向にバイアスする
    # return onsets[onsets <= _T]
    onsets = onsets[(onsets > 10) & (onsets <= _T + 10)] - 10
    adjust = (_T - (onsets[0] + onsets[-1])) / 2
    onsets = (onsets + adjust).astype(np.int64)
    return onsets[onsets <= _T]


def generate_pulse():
    return np.ones((_PULSE_DURATION, 3, 3))

def generate_visual_flickr(freq: int, d: int = _PULSE_DURATION, t: int = _T, n: int = _N, shift=None):
    npulse = freq * int(_TAU * t / 1000)
    flickr = np.zeros((t, n, n))
    pulse = np.zeros(t).astype(int)
    if shift is None:
        shift = 0
    if freq > 0:
        onsets = generate_pulse_onsets(freq, shift=shift)
    else:
        onsets = []
    for onset in onsets:
        flickr[(onset-1), 4:7, 4:7] = generate_pulse()
        pulse[(onset-1)] = 1
    return flickr, pulse

##############################
# Audiotory flickr generator #
##############################

_FS = 48000
_NFRAME_10MILIS = int(_FS / 1000 * _PULSE_DURATION)
_ONSET = 53250

clicks = np.array(AudioSegment.from_file("./stim/click.m4a", "m4a").get_array_of_samples())

click = clicks[_ONSET:_ONSET + _NFRAME_10MILIS]
empty = clicks[_FS:_FS + _NFRAME_10MILIS]

def fft(wave, outdim: int = _N * _N, fs: int = _FS):
    wave_fft = np.fft.fft(wave)
    freq = np.fft.fftfreq(wave.size, 1/fs)[1:int(wave_fft.size/2)].astype(np.int64)
    amp = abs(wave_fft/(wave_fft.size/2))[1:int(wave_fft.size/2)]
    idx = np.linspace(1, amp.size-1, outdim).astype(np.int64)
    return freq[idx], amp[idx] / np.max(amp[idx])

def add_noise(wave, alpha):
    return wave + alpha * np.random.normal(0, np.std(wave), wave.size)

def generate_audio_flickr(freq: int, d: int = _PULSE_DURATION, t: int = _T, n: int = _N, shift=None):
    audio_flickr = []
    pulse = np.zeros(t).astype(int)
    for _ in range(_T):
        _, amp = fft(add_noise(empty, 0.00))
        amp = np.random.uniform(0, 0.00, len(amp))
        audio_flickr.append(amp)
    if shift is None:
        shift = 0
    if freq > 0:
        onsets = generate_pulse_onsets(freq, shift=shift)
    else:
        onsets = []
    for onset in onsets:
        _, amp = fft(add_noise(click, 0.00))
        audio_flickr[(onset-1)] = amp
        pulse[(onset-1)] = 1
    return np.array(audio_flickr), pulse


####################
# Reward generator #
####################
def generate_reward(freq: int, threshold: int = 9):
    if freq > threshold:
        return np.ones(T)
    return np.zeros(T)

def generate_synchronous_dataset(freq: list[int], rep: int, vsft=None, asft=None):
    freq_ = freq * rep
    return [(generate_visual_flickr(f, shift=vsft), generate_audio_flickr(f, shift=asft), generate_reward(f), f, f) for f in freq_]

def generate_visual_dataset(freq: list[int], rep: int, vsft=None):
    freq_ = freq * rep
    return [(generate_visual_flickr(f, shift=vsft), generate_audio_flickr(0), generate_reward(f), f, 0) for f in freq_]

def generate_audio_dataset(freq: list[int], rep: int, asft=None):
    freq_ = freq * rep
    return [(generate_visual_flickr(0), generate_audio_flickr(f, shift=asft), generate_reward(0), 0, f) for f in freq_]

def generate_asynchronous_dataset(freq: list[int], rep: int, vsft=None, asft=None):
    freq_ = freq * rep
    return [(generate_visual_flickr(9, shift=vsft), generate_audio_flickr(f, shift=asft), generate_reward(9), 9, f) for f in freq_]

def gaussian_smoothing(dataset, sigma_v, sigma_a):
    if sigma_v > 0 and sigma_a > 0:
        return [((gaussian_filter1d(v, sigma=sigma_v, axis=0, mode="constant"), vp), (gaussian_filter1d(a, sigma=sigma_a, axis=0, mode="constant"), ap), r, vl, al) for ((v, vp), (a, ap), r, vl, al) in dataset]
    elif sigma_v > 0 and sigma_a <= 0:
        return [((gaussian_filter1d(v, sigma=sigma_v, axis=0, mode="constant"), vp), (a, ap), r, vl, al) for ((v, vp), (a, ap), r, vl, al) in dataset]
    elif sigma_v <= 0 and sigma_a > 0:
        return [((v, vp), (gaussian_filter1d(a, sigma=sigma_a, axis=0, mode="constant"), ap), r, vl, al) for ((v, vp), (a, ap), r, vl, al) in dataset]
    else:
        return dataset

########################
# Generate lagged data #
########################

def generate_lagged_visual_flickr(freq: int, d: int = _PULSE_DURATION, t: int = _T, n: int = _N, lag=0):
    npulse = freq * int(_TAU * t / 1000)
    flickr = np.zeros((t, n, n))
    pulse = np.zeros(t).astype(int)
    if freq > 0:
        onsets = generate_pulse_onsets(freq, shift=0)
        max_lag = np.median(np.diff(onsets)) - 1
        if abs(lag) > max_lag:
            lag = max_lag if lag >= 0 else -max_lag
        lag_idx = np.argmax(np.diff(onsets > 50)) + 1
        onsets[lag_idx] += lag
    else:
        onsets = []

    for onset in onsets:
        flickr[onset:(onset+_PULSE_DURATION), 4:7, 4:7] = generate_pulse()
        pulse[onset:(onset+d)] = np.ones(d)
    return flickr

def generate_lagged_audio_flickr(freq: int, d: int = _PULSE_DURATION, t: int = _T, n: int = _N, lag=0):
    audio_flickr = []
    pulse = np.zeros(t).astype(int)
    for _ in range(_T):
        _, amp = fft(add_noise(empty, 0.00))
        amp = np.random.uniform(0, 0.00, len(amp))
        audio_flickr.append(amp)
    if freq > 0:
        onsets = generate_pulse_onsets(freq, shift=0)
        max_lag = np.median(np.diff(onsets)) - 1
        if abs(lag) > max_lag:
            lag = max_lag if lag < 0 else -max_lag
        lag_idx = np.argmax(np.diff(onsets > 50)) + 1
        onsets[lag_idx] += lag
    else:
        onsets = []

    for onset in onsets:
        for i in range(d):
            _, amp = fft(add_noise(click, 0.00))
            audio_flickr[onset+i] = amp
            pulse[onset+i] = 1
    return np.array(audio_flickr)

# def generate_dropped_visual_flickr(freq: int, d: int = _PULSE_DURATION, t: int = _T, n: int = _N, drop: bool = True):
#     npulse = freq * int(_TAU * t / 1000)
#     flickr = np.zeros((t, n, n))
#     if freq > 0:
#         onsets = generate_pulse_onsets(freq, shift=0)
#         drop_idx = np.argmax(np.diff(onsets > 50)) + 1
#         if drop:
#             onsets = np.delete(onsets, drop_idx)
#     else:
#         onsets = []
# 
#     for onset in onsets:
#         flickr[onset:(onset+_PULSE_DURATION), 4:7, 4:7] = generate_pulse()
#     return flickr
# 
# def generate_dropped_audio_flickr(freq: int, d: int = _PULSE_DURATION, t: int = _T, n: int = _N, drop: bool = True):
#     audio_flickr = []
#     for _ in range(_T):
#         _, amp = fft(add_noise(empty, 0.00))
#         amp = np.random.uniform(0, 0.00, len(amp))
#         audio_flickr.append(amp)
#     if freq > 0:
#         onsets = generate_pulse_onsets(freq, shift=0)
#         drop_idx = np.argmax(np.diff(onsets > 50)) + 1
#         if drop:
#             onsets = np.delete(onsets, drop_idx)
#     else:
#         onsets = []
# 
#     for onset in onsets:
#         for i in range(d):
#             _, amp = fft(add_noise(click, 0.00))
#             audio_flickr[onset+i] = amp
#     return np.array(audio_flickr)

def generate_lagged_visual_dataset(freq: list[int], rep: int, lag: int):
    freq_ = freq * rep
    return [(generate_lagged_visual_flickr(f, lag=lag), generate_lagged_audio_flickr(f, lag=0), "visual", f, lag) for f in freq_]

def generate_lagged_audio_dataset(freq: list[int], rep: int, lag: int):
    freq_ = freq * rep
    return [(generate_lagged_visual_flickr(f, lag=0), generate_lagged_audio_flickr(f, lag=lag), "audio", f, lag) for f in freq_]

def generate_lagged_audvis_dataset(freq: list[int], rep: int, lag: int):
    freq_ = freq * rep
    return [(generate_lagged_visual_flickr(f, lag=lag), generate_lagged_audio_flickr(f, lag=lag), "audvis", f, lag) for f in freq_]

# def generate_dropped_visual_dataset(freq: list[int], rep: int):
#     freq_ = freq * rep
#     return [(generate_dropped_visual_flickr(f, drop=True), generate_dropped_audio_flickr(f, drop=False), "visual", f) for f in freq_]
# 
# def generate_dropped_audio_dataset(freq: list[int], rep: int, lag: int):
#     freq_ = freq * rep
#     return [(generate_dropped_visual_flickr(f, drop=False), generate_dropped_audio_flickr(f, drop=True), "audio", f) for f in freq_]
# 
# def generate_dropped_audvis_dataset(freq: list[int], rep: int, lag: int):
#     freq_ = freq * rep
#     return [(generate_dropped_visual_flickr(f, drop=True), generate_dropped_audio_flickr(f, drop=True), "audvis", f) for f in freq_]
