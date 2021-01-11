#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:56:29 2020

@author: mpolonenko
"""
import numpy as np
import gc
from mne.filter import resample
from expyfun.io import read_hdf5, write_hdf5
import scipy.signal as sig
from pyfftw.interfaces.scipy_fftpack import fft as fftw, ifft as ifftw
import matplotlib.pyplot as plt


def fft(*args, **kwargs):
    return fftw(*args, threads=11, **kwargs)


def ifft(*args, **kwargs):
    return ifftw(*args, threads=11, **kwargs)


# %% input subject and experiment information
n_jobs = 'cuda'

# saving options
overwrite_file = True

# paths and root filenames
opath = '/mnt/data2/paper_peaky_speech/'
data_path = opath + 'subject_data/clockadjust_use/exp1/'
save_path = opath + 'analysis/model/exp1/'

# %% setup parameters
# stimulus
narrator = ['Male']
task = 'peakyvsunaltered'
stimuli = ['broadband', 'multiband']
stim_index = {si: i for i, si in enumerate(stimuli)}
audio = ['x_play_single', 'x_play_band']
audio_index = {st: aud for aud, st in zip(audio, stimuli)}

# other parameters
band_freqs = ['0-1 kHz', '1-2 kHz', '2-4 kHz', '4-8 kHz']
n_bands = len(band_freqs)
fs = 10e3
len_stim = 64.

# %% load stim and eeg
print('=== loading responses ===')
data = read_hdf5(data_path + task + '_responses.hdf5')
w_pulses = data['w_pulses'][:, 1:]
w_subtr = data['w_subtr']
del data

n_stim = len(stimuli)
n_subs = w_pulses.shape[0]
n_nar = len(narrator)

b, a = sig.butter(1, 30. / (fs / 2.), btype='highpass')
w_pulses = np.fft.fftshift(sig.lfilter(b, a, np.fft.ifftshift(
    w_pulses, -1), -1), -1)
w_subtr = np.fft.fftshift(sig.lfilter(b, a, np.fft.ifftshift(
    w_subtr, -1), -1), -1)

# %% import stim and resample
print('=== loading and resampling stimuli ===')
data = read_hdf5(data_path + task + '_stimuli.hdf5')
audio = data['audio'][1:]
pulseinds = data['pulseinds'][1:]
fs_stim = data['fs_stim']
del data
gc.collect()

rectaudio = [np.maximum(0, audio), np.maximum(0, -audio)]
del audio
gc.collect()
# (stim, chap, pos/neg, samples)
rectaudio = np.array(rectaudio).transpose(1, 2, 0, 3)
rectaudio = resample(rectaudio, int(fs * len_stim), rectaudio.shape[-1],
                     npad='auto', n_jobs=n_jobs)

pulsetrains = []
for si in range(2):
    pulsetrain = []
    for pulseind in pulseinds[si]:
        pulses = np.zeros((len(pulseind), int(fs * len_stim)))
        pinds = [(pi * float(fs) / fs_stim).astype(int) for pi in pulseind]
        for bi in range(len(pinds)):
            pulses[bi, pinds[bi]] = 1.
        pulsetrain += [pulses]
        del pulses
    pulsetrains += [np.array(pulsetrain)]
del pulsetrain
del pulseinds

n_chaps = rectaudio.shape[2]

# %% create ABR kernels
print('=== creating kernels ===')


def make_kernel(kn, shift):
    for i, si in enumerate(shift):
        fix = kn[i].copy()
        fix -= fix[[si]]
        fix[:si] = 0
        fix = np.pad(fix, (int(3 * si), si), mode='constant')
        fix = fix * sig.tukey(fix.shape[-1], 1)
        fix = fix[int(3 * si):-si]
        kn[i] = fix
        del fix
    plt.plot(kn.T)
    kn /= kn.max(-1).max(-1)
    kn = np.pad(kn, ((0, 0), (kn.shape[-1], 0)), mode='constant')
    # plt.plot(kn.T)
    return kn


tinds_kernel = np.arange(0, int(16e-3 * fs))

wf = w_pulses.mean(0)[stim_index['broadband'], tinds_kernel][np.newaxis]
# plt.plot(wf.T)
shifts = [int(1.6e-3 * fs)]
# plt.clf()
# plt.plot((wf[0] - wf[0][[shifts[0]]]).T)
kernel_broadband = make_kernel(wf, shifts)
plt.clf()
plt.plot(kernel_broadband.T)

plt.clf()
wf = w_subtr.mean(0)[:, tinds_kernel]
# plt.plot(wf.T)
shifts = [35, 30, 25, 20]
kernel_multiband = make_kernel(wf, shifts)
plt.clf()
plt.plot(kernel_multiband[:, len(tinds_kernel):].T)

del wf
# %% deconvolve
print('==== deconvolving ===')

kernels = [kernel_broadband, kernel_multiband]

w_linear = []
w_common = []
for si, ki in enumerate(kernels):
    fake_eeg = np.array([
        sig.fftconvolve(rectaudio[si],
                        np.tile(ki[i], (n_chaps, 2, 1)),
                        'same') for i in range(ki.shape[-2])]).transpose(
                            1, 0, 2, 3)

    fft_x = fft(pulsetrains[si])[:, 0, np.newaxis, np.newaxis, :]
    fft_y = fft(fake_eeg)
    w = ifft((np.conj(fft_x) * fft_y).mean(0) /
             (np.conj(fft_x) * fft_x).mean(0)).real
    w = w.mean(-2)  # average across pos/neg
    w_linear += [w]
    del w, fft_x, fft_y

    fft_x = fft(pulsetrains[si])[:, -1, np.newaxis, np.newaxis, :]
    fft_y = fft(fake_eeg)
    w = ifft((np.conj(fft_x) * fft_y).mean(0) /
             (np.conj(fft_x) * fft_x).mean(0)).real
    w = w.mean(-2)
    w_common += [w]
    del w, fft_x, fft_y


# %% save
print('=== saving ===')
write_hdf5(save_path + 'exp1_linear_model_data.hdf5', dict(
    kernels_avg=[kernel_broadband, kernel_multiband],
    w_linear=w_linear,
    w_common=w_common), overwrite=True)
