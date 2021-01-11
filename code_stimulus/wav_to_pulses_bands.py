#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on Wed Sep 23 18:13:18 2020
@author: rkmaddox; mpolonenko

"""

import numpy as np
from expyfun.io import read_wav, write_wav, write_hdf5
from glob import glob
import tempfile
import shutil
import os
from scipy.fftpack import ifft, ifftshift
from scipy.interpolate import interp1d, RegularGridInterpolator
import scipy.signal as sig
import parselmouth
import matplotlib.pyplot as plt
plt.ion()

# %% User settings

band_type = 'original'  # original (4-band) or audiological (5-band)
n_ears = 1  # 1 (diotic) or 2 (dichotic)

save_unaltered = False
save_broadband = True
save_multiband = True

overwrite_file = True
save_hdf5 = True

story = 'alchemyst'  # alchemyst (male narrator) or wrinkle (female narrator)
main_path = '/mnt/data/abr_peakyspeech/'
wav_in_path = main_path + 'audio_books/{}/'.format(story)
out_path = '/mnt/data/abr_peakyspeech/{}/'.format(story)

start_file = 0
n_trials = 120

# %% Set up the bands and f_shifts

fs_filt = 44100
n_filt = int(fs_filt * 5e-3)
n_filt += ((n_filt + 1) % 2)  # must be odd
freq = np.arange(n_filt) / float(n_filt) * fs_filt

# also determines HP cutoff for mixing high-freq original speech (8 kHz)
fc_band_ori = 1e3 * 2 ** np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
fc_band_aud = 1e3 * 2. ** np.array([-1, 0, 1, 2, 3, 4])

if band_type == 'original':
    fc_band = fc_band_ori
    name_band = ''
else:
    fc_band = fc_band_aud
    name_band = '_aud'

if n_ears == 1:
    name_ears = '_diotic'
    f0_band = fc_band
else:
    name_ears = '_dichotic'
    f0_band = np.repeat(fc_band, n_ears)

n_band = len(fc_band) - 1
n_f0 = n_ears * n_band  # would just be n_ears for only broadband
n_f0_fake = 6  # number of fake pulse trains to calculate common component

# create a list of prime numbers
rmin = 2
rmax = 200
check = range(rmin, rmax)
primes = []
for i in check:
    for j in range(rmin, i):
        if not i % j:
            break
    else:
        primes.append(i)
primes = primes[1:]  # didn't originally use 2
del primes[8]  # note: when making original files, 29 wasn't included
primes = np.array(primes)

f_shifts = 0.5 * (primes ** 0.5 - 1)  # in Hz
f_shift_band = np.append(f_shifts[:n_f0 - 1 + int(n_f0_fake / 2)],
                         -f_shifts[:int(n_f0_fake / 2)])

assert(len(f_shift_band) == n_f0 - 1 + n_f0_fake)  # left the first unshifted

# %% Make band filters


def amp_fun(f, top_width, trans_width):
    # f, top_width, trans_width all assume OCTAVES
    # top and trans width should add up to 1 for complementary filters
    return np.maximum(0, np.minimum(
            1, 1 + top_width / (2 * trans_width) - np.abs(f) / trans_width))


top_width = 0.5
trans_width = 0.5

amp_band = np.zeros((n_band + 1, n_filt))
for fi, f in enumerate(fc_band):
    amp_band[fi] = amp_fun(np.log2(freq / f), top_width, trans_width) ** 0.5
amp_band[0, freq <= fc_band[0]] = 1
amp_band[-1, freq >= fc_band[-1]] = 1
amp_band[:, :-n_filt // 2:-1] = amp_band[:, 1:n_filt // 2 + 1]

h_band = ifftshift(ifft(amp_band).real, axes=-1) * \
    np.atleast_2d(sig.nuttall(n_filt))

# Make single-band filter - based off the  multiband filters
amp_single = np.copy(amp_band[-2:])
amp_single[0, freq <= fc_band[-2]] = 1
amp_single[:, :-n_filt // 2:-1] = amp_single[:, 1:n_filt // 2 + 1]
h_single = ifftshift(ifft(amp_single).real, axes=-1) * \
    np.atleast_2d(sig.nuttall(n_filt))


# plot broadband filters
plt.figure(figsize=(3.5, 4))
plt.subplot(311)
plt.semilogx(freq, amp_single.T, '-')
plt.semilogx(freq, (amp_band ** 2).sum(0), 'k:')
plt.xlim([1, 12e3])
plt.ylim([0, 1.1])
plt.ylabel('Gain')
plt.title('Design frequency response')
plt.subplot(312)
plt.plot((np.arange(n_filt) - n_filt // 2) / float(fs_filt) * 1e3, h_single.T)
plt.xlabel('Time (ms)')
plt.title('Impulse response (windowed)')
plt.subplot(313)
plt.semilogx(freq, 20 * np.log10(np.abs(np.fft.fft(h_single))).T, '-')
plt.semilogx(freq, 10 * np.log10((np.abs(np.fft.fft(h_band)) ** 2).sum(0)),
             'k:')
plt.xlim([0.1, 12e3])
plt.ylim([-60, 6])
plt.ylabel('Gain (dB)')
plt.title('Actual frequency response')
plt.tight_layout(0.2)
plt.savefig(out_path + 'filters_broadband' + name_band + '.png',
            overwrite=overwrite_file)
plt.close()

# plot multiband filters
plt.figure(figsize=(3.5, 4))
plt.subplot(311)
plt.semilogx(freq, amp_band.T)
plt.semilogx(freq, (amp_band ** 2).sum(0), 'k:')
plt.xlim([0, 12e3])
plt.ylim([0, 1.1])
plt.ylabel('Gain')
plt.title('Design frequency response')
plt.subplot(312)
plt.plot((np.arange(n_filt) - n_filt // 2) / float(fs_filt) * 1e3, h_band.T)
plt.xlabel('Time (ms)')
plt.title('Impulse response (windowed)')
plt.subplot(313)
plt.semilogx(freq, 20 * np.log10(np.abs(np.fft.fft(h_band))).T)
plt.semilogx(freq, 10 * np.log10((np.abs(np.fft.fft(h_band)) ** 2).sum(0)),
             'k:')
plt.xlim([0.1, 12e3])
plt.ylim([-60, 6])
plt.ylabel('Gain (dB)')
plt.title('Actual frequency response')
plt.tight_layout(0.2)
plt.savefig(out_path + 'filters_multiband' + name_band + '.png',
            overwrite=overwrite_file)
plt.close()

# %% make the stimuli

files = sorted(glob(wav_in_path + story + '*.wav'))
if story == 'alchemyst':
    f0_min, f0_max = 60, 350
elif story == 'wrinkle':
    f0_min, f0_max = 90, 500
slop = 1.6
n_ep_max = 120

for fi, fn in enumerate(files[start_file:n_trials]):
    print('\n===== %s =====\n' % fn.split('/')[-1])
    print('analyzing pulses')
    temp_dir = tempfile.mkdtemp(prefix='wav_to_pulses')
    try:
        shutil.copyfile(fn, temp_dir + '/stim.wav')
        # call the praat processing
        wav_path = temp_dir + '/stim'
        parselmouth.praat.run_file('wav_to_pulses.praat', '%s' % wav_path,
                                   str(f0_min), str(f0_max))
        # read in the pulse times
        with open(wav_path + '.PointProcess') as fid:
            lines = np.copy(fid.readlines())
        assert len(lines) > 7
        pulse_times = np.array([float(l.replace(' ', '').strip().split('=')[1])
                                for l in lines[7:]])
    except Exception:
        shutil.rmtree(temp_dir)
        raise Exception
    shutil.rmtree(temp_dir)
    x, fs = read_wav(fn)
    assert(fs == fs_filt)
    x = x[0]
    x *= 0.01 / x.std()  # normalize to rms = 0.01

    # %% smooth the pulses
    pulse_times_fix = np.copy(pulse_times)
    n_smooth_iter = 10
    for _ in range(n_smooth_iter):
        pulse_times_fix_start = np.copy(pulse_times_fix)
        for pii in np.arange(1, len(pulse_times) - 1):
            if np.abs(np.log2(np.diff(pulse_times_fix_start[[pii - 1, pii]])) -
                      np.log2(np.diff(pulse_times_fix_start[[pii, pii + 1]])))\
                    < np.log2(slop):
                pulse_times_fix[pii] = np.mean(
                    pulse_times_fix_start[pii - 1:pii + 1 + 1])
    pulse_inds = np.round(pulse_times_fix * fs).astype(int)
    pulse_inds_unfix = np.round(pulse_times * fs).astype(int)

    # %% find the gaps for sign reversals (helps mitigate artifact)
    b_env, a_env = sig.butter(1, 6 / (fs / 2.))
    env = sig.filtfilt(b_env, a_env, np.abs(x))
    flip_regions = np.where(env < .01 * np.median(env))[0]
    flip_inds = flip_regions[np.where(np.diff(flip_regions) > 1)[0] - 1]
    flip_spikes = np.zeros(env.shape)
    flip_spikes[flip_inds] = 1
    b_smooth, a_smooth = sig.butter(1, 10e3 / (fs / 2.))
    flip_sign = sig.filtfilt(b_smooth, a_smooth,
                             (-1) ** (np.cumsum(flip_spikes) +
                                      files.index(fn) % 2))

    # %% analyze the spectrogram and create mixer (when voiced/voiceless)
    print('analyzing spectrogram')
    nperseg = int(fs / 50.)  # denominator is freq in Hz
    noverlap = nperseg - int(fs / f0_max)
    f_sg, t_sg, sg = sig.spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap,
                                     scaling='spectrum', window='hamming')
    phase = np.nan * np.ones((n_f0 + n_f0_fake, x.shape[-1]))
    breaks = np.where(np.diff(pulse_times_fix) >= 1. / f0_min)[0]

    mixer = np.zeros(x.shape)
    for bii in range(len(breaks) - 1):
        inds = np.arange(pulse_inds[breaks[bii] + 1],
                         pulse_inds[breaks[bii + 1] - 1])
        if len(inds):
            try:
                ifun = interp1d(pulse_inds[breaks[bii] + 1:breaks[bii + 1]],
                                2 * np.pi * np.arange(breaks[bii + 1] -
                                                      breaks[bii] - 1),
                                kind='cubic')
                print('cubic', len(inds))
            except:
                try:
                    ifun = interp1d(pulse_inds[breaks[bii] +
                                               1:breaks[bii + 1]],
                                    2 * np.pi * np.arange(breaks[bii + 1] -
                                                          breaks[bii] - 1),
                                    kind='quadratic')
                    print('quadratic', len(inds))
                except:
                    ifun = interp1d(pulse_inds[breaks[bii] +
                                               1:breaks[bii + 1]],
                                    2 * np.pi * np.arange(breaks[bii + 1] -
                                                          breaks[bii] - 1),
                                    kind='linear')
                    print('linear', len(inds))
            phase[0, inds] = ifun(inds)
        if len(inds) > 1:
            start_inds = np.arange(pulse_inds[breaks[bii] + 1],
                                   pulse_inds[breaks[bii] + 2])
            stop_inds = np.arange(pulse_inds[breaks[bii + 1] - 2],
                                  pulse_inds[breaks[bii + 1] - 1])
            mixer[start_inds] = np.sin(np.arange(len(start_inds)) * 0.5 *
                                       np.pi / len(start_inds)) ** 2
            mixer[stop_inds] = np.sin(np.arange(len(stop_inds)) * 0.5 *
                                      np.pi / len(stop_inds))[::-1] ** 2
            mixer[start_inds[-1]:stop_inds[0]] = 1
    mixer = np.atleast_2d(1 - mixer)

    xsg, ysg = np.meshgrid(t_sg, f_sg)
    interp = RegularGridInterpolator([f_sg, t_sg], sg, method='linear',
                                     bounds_error=False, fill_value=0)

    # %% make the new band phases and pulse_inds
    print('make new band phases and pulse_inds')
    for f0_ind in np.arange(1, n_f0 + n_f0_fake):
        phase[f0_ind] = phase[0] + f_shift_band[f0_ind - 1] * (
                2 * np.pi * np.arange(phase.shape[-1]) / float(fs))
        # + np.random.rand() * 2 * np.pi)  # random start phase for future
    f0 = np.diff(phase, axis=-1) * fs / 2 / np.pi
    t0 = np.arange(phase.shape[-1] - 1) / float(fs)

    f_harm_max = np.minimum(fc_band[-1] * 2, fs / 2. - 1)
    n_harm = int(np.floor(f_harm_max / f0_min))
    amp0 = np.zeros((n_harm, len(x)))

    for hi in range(n_harm):
        # rather than interp at every point, take the pulse points and then
        # cubic spline interpolate each of those
        amp0[hi, :-1] = interp(np.transpose([f0[0] * (hi + 1), t0])) ** 0.5
    amp0[:, np.isnan(amp0[0])] = 0
    bplp, aplp = sig.butter(1, 70 / (fs / 2.))
    amp0 = np.array([sig.filtfilt(bplp, aplp, amp) for amp in amp0])

    print('Generating the harmonics')
    x_harm = np.zeros((n_f0, x.shape[-1]))  # only for real pulse trains
    for f0_ind in range(n_f0):
        for hi in range(n_harm):
            x_harm[f0_ind] += np.nan_to_num(np.cos(phase[f0_ind] * (hi + 1))
                                            * amp0[hi])
        """
        MODIFY THE PHASE BY A NONLINEAR FUNCTION THAT SHIFTS when
        f0 * hi + 1 is higher than a certain cutoff, e.g. 2k. this where there
        should be no beating, and the frequency transition can be broad enough
        to avoid super nasty phase skip artifacts--need to be careful about
        pulse times for reconstruction though...
        """
        x_harm[f0_ind, np.isnan(x_harm[f0_ind])] = 0
    x_harm *= (1 - mixer)
    x_harm *= ((1 - mixer) * x).std() / x_harm.std()
    x_harm *= -1
    x *= -1  # this is to make it the same sign (cond) as x_harm (inspection)

    # filter the different f0_bands
    x_harm_band = np.zeros((n_band, n_ears, len(x)))
    f0_ind = 0
    for band_ind in range(n_band):
        for ear_ind in range(n_ears):
            x_harm_band[band_ind, ear_ind] = sig.fftconvolve(x_harm[f0_ind],
                                                             h_band[band_ind],
                                                             'same')
            f0_ind += 1
    if n_ears == 1:
        x_harm_band = x_harm_band.mean(1)
    # %% combine the different bands, as well as voiced/voiceless segments

    x_harm_mix = x_harm_band.sum(0)

    x_play_band = (mixer[0] * x + x_harm_mix + (1 - mixer[0]) *
                   sig.fftconvolve(x, h_band[-1], 'same'))
    if n_ears == 1:
        x_play_single = (mixer[0] * x + sig.fftconvolve(x_harm[0],
                                                        h_single[0], 'same') +
                         (1 - mixer[0]) * sig.fftconvolve(x, h_single[1],
                                                          'same'))
    else:
        x_play_single = (mixer[0] * x + sig.fftconvolve(
            x_harm[:2], np.atleast_2d(h_single[0]), 'same') + (1 - mixer[0]) *
            sig.fftconvolve(x, h_single[1], 'same'))

    pulse_inds = [np.where(np.diff(np.mod(phase[bi], 2 * np.pi)) < 0)[0]
                  for bi in range(n_f0 + n_f0_fake)]

    x_play_band *= flip_sign
    x_play_single *= flip_sign
    x *= flip_sign

    # %% save files
    print('saving')
    save_paths = [out_path + '{}_wav/multiband/'.format(story),
                  out_path + '{}_wav/broadband/'.format(story),
                  out_path + '{}_wav/unaltered/'.format(story),
                  out_path + '{}_hdf5/hdf5_full/'.format(story),
                  out_path + '{}_hdf5/hdf5_reduced/'.format(story)]
    for sp in save_paths:
        if not os.path.exists(sp):
            os.makedirs(sp)

    if save_multiband:
        write_wav(out_path + '{}_wav/multiband/'.format(story) +
                  fn.split('/')[-1][:-4] + '_multiband' + name_band +
                  name_ears + fn.split('/')[-1][-4:], x_play_band, fs,
                  overwrite=overwrite_file)
    if save_broadband:
        write_wav(out_path + '{}_wav/broadband/'.format(story) +
                  fn.split('/')[-1][:-4] + '_broadband' + name_band +
                  name_ears + fn.split('/')[-1][-4:], x_play_single, fs,
                  overwrite=overwrite_file)
    if save_unaltered:
        write_wav(out_path + '{}_wav/unaltered/'.format(story) +
                  fn.split('/')[-1][:-4] + '_unaltered' + name_ears +
                  fn.split('/')[-1][-4:], x, fs, overwrite=overwrite_file)

    if save_hdf5:
        write_hdf5(out_path + '{}_hdf5/hdf5_full/'.format(story) +
                   fn.split('/')[-1][:-4] + '_bands' + name_band + '.hdf5',
                   dict(
            pulse_inds=pulse_inds,
            pulse_inds_unfix=pulse_inds_unfix,
            mixer=mixer,
            x_harm_band=x_harm_band,
            x_harm_mix=x_harm_mix,
            x_harm=x_harm,
            x_play_band=x_play_band,
            x_play_single=x_play_single,
            x=x,
            fs=fs,
            f0_min=f0_min,
            f0_max=f0_max,
            n_smooth_iter=n_smooth_iter,
            fc_band=fc_band,
            amp_band=amp_band,
            amp_single=amp_single,
            h_band=h_band,
            h_single=h_single,
            f_harm_max=f_harm_max,
            n_harm=n_harm,
            f_shift_band=f_shift_band,
            n_f0=n_f0,
            n_f0_fake=n_f0_fake,
            slop=slop,
            top_width=top_width,
            trans_width=trans_width,
            n_filt=n_filt,
            phase=phase), overwrite=overwrite_file)
        write_hdf5(out_path + '{}_hdf5/hdf5_reduced/'.format(story) +
                   fn.split('/')[-1][:-4] + '_bands' + name_band +
                   '_reduced.hdf5', dict(
            pulse_inds=pulse_inds,
            mixer=mixer,
            x_play_band=x_play_band,
            x_play_single=x_play_single,
            x=x,
            fs=fs,
            fc_band=fc_band), overwrite=overwrite_file)