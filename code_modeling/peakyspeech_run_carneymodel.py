#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:05:48 2020

@author: mpolonenko
"""
from mne.filter import resample
from expyfun.io import read_hdf5, write_hdf5
import numpy as np
import ic_cn2018 as nuclei
import cochlea
from joblib import Parallel, delayed
import datetime
from scipy.fftpack import fft, ifft
import gc
import time

'''
NOTE: Must download the following code:
Code: 2018 Model: Cochlea+OAE+AN+ABR+EFR (Matlab/Python) from
    https://www.waves.intec.ugent.be/members/sarah-verhulst
UR_EAR_2020b from https://www.urmc.rochester.edu/labs/carney.aspx
'''
# %% define functions


def doanmodel(ti, cfi):
    stim_up = db_conv * ti  # scale intensity

    # filter parameters
    cohc = 1.
    cihc = 1.

    if do_IHC:  # get IHC response only
        anf_rates_up = cochlea.zilany2014._zilany2014.run_ihc(
            stim_up,
            cf[cfi],
            fs_up,
            species='human',
            cohc=cohc,
            cihc=cihc
        )
    else:  # get anf rate response
        anf_rates_up = cochlea.run_zilany2014_rate(
            stim_up,
            fs_up,
            anf_types=anf_types,
            cf=cf[cfi],
            species='human',
            cohc=cohc,
            cihc=cihc
        )

    anf_rates = resample(anf_rates_up.T, fs_out, fs_up, npad='auto',
                         n_jobs=1)
    anf_rates = anf_rates.reshape(len(anf_types), anf_rates.shape[-1])
    return anf_rates


def dobsmodel(response):
    anfH = response[:, 0].T
    anfM = np.zeros(anfH.shape)
    anfL = np.zeros(anfH.shape)

    # get CN response from AN response
    cn, anSummed = nuclei.cochlearNuclei(anfH, anfM, anfL, numH,
                                         numM, numL, fs_out)
    # get IC response from CN response
    ic = nuclei.inferiorColliculus(cn, fs_out)

    # scale responses and sum over CF
    w1 = nuclei.M1*np.sum(anSummed, axis=1)
    w3 = nuclei.M3*np.sum(cn, axis=1)
    w5 = nuclei.M5*np.sum(ic, axis=1)

    # lag the responses appropriately
    w1_shifted = np.roll(w1, int(fs_out / 1e3), axis=0)
    w3_shifted = np.roll(w3, int(2.25 * fs_out / 1000), axis=0)
    w5_shifted = np.roll(w5, int(3.5 * fs_out / 1000), axis=0)

    # sum the responses together to get abr
    summed = w1_shifted + w3_shifted + w5_shifted
    return summed.T, w1_shifted.T, w3_shifted.T, w5_shifted.T


# %% model parameters
anf_types = ['hsr']  # select lsr, msr, or hsr for spont rate
dOct = 1. / 6  # Cfs in sixth octaves 1./6
cf = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), dOct)
n_jobs = 10  # use 10 cores of cpu (could be -1 to use all but 1)

# Number of low, medium, and high spont. rate fibers
numL = 0
numM = 0
numH = len(cf)

do_IHC = False

stim_db = 65
stim_rms = 0.01
sine_rms_at_0db = 20e-6
# scalar to put it in units of pascals (double-checked and good)
db_conv = ((sine_rms_at_0db / stim_rms) * 10 ** (stim_db / 20.))

fs_up = 100e3  # carney model requires high sampling rate
fs_out = 10e3

# %% stim parameters
opath = '/mnt/data2/paper_peaky_speech/'
stim_path = opath + 'open_access/Dryad/{}_Stimuli_{}_Narrator/'
save_path = opath + 'analysis/model/{}/'

narrators = ['Male', 'Female']
stim_root = '{}Narrator{:03}_bands.hdf5'
save_root_model = '{}_{}_{}Narrator_model.hdf5'
save_root_stim = '{}_{}_{}Narrator_stimuli.hdf5'
save_root_responses = '{}_{}_{}Narrator_responses.hdf5'


bands = ['0-1 kHz', '1-2 kHz', '2-4 kHz', '4-8 kHz']
bands_aud = ['.5 kHz', '1 kHz', '2 kHz', '4 kHz', '8 kHz']

stimuli = ['x_play_single', 'x_play_band', 'x']
stimulus_index = {st: i for i, st in enumerate(stimuli)}
len_stim = 64
n_trials_total = 120
chapts_exp1 = [np.arange(1, n_trials_total, 3),
               np.arange(2, n_trials_total, 3),
               np.arange(0, n_trials_total, 3)]
chapts_exp2 = [np.arange(0, 60, 2), np.arange(1, 60, 2)]
chapts_exp3 = [np.arange(0, 60, 1), np.arange(0, 60, 1)]
chapts_pilot = [np.arange(15), np.arange(60)]

s_sleep = 3
n_jobs = 'cuda'

scale = 401 / len(cf)  # recommended by verhulst for abr amplitude scaling

# %% STIMULUS TO RUN [modify here only]
exp = 'Exp1'  # options: Exp1, Exp2, Exp3, Pilot
stimulus = 'x_play_band'  # options: x, x_play_single, x_play_band
chapt = chapts_exp1[stimulus_index[stimulus]]
nar = narrators[0]
fs_in = 44100
# fs_in = 48e3  # for pilot only

# %%
if stimulus == 'x_play_band':
    n_bands = len(bands)
    n_fake = n_bands
else:
    n_bands = 1
    n_fake = 0
n_bands_all = n_bands + n_fake
n_trials = len(chapt)

# %% run the models
print('\n== {} ==='.format(stimulus))
print('== running model ==')

an_mod_all = np.zeros((n_trials, len(cf), len_stim * int(fs_out)))
bs_mod_all = np.zeros((n_trials, len_stim * int(fs_out)))
w1_mod_all = np.zeros((n_trials, len_stim * int(fs_out)))
w3_mod_all = np.zeros((n_trials, len_stim * int(fs_out)))
w5_mod_all = np.zeros((n_trials, len_stim * int(fs_out)))

for i, ci in enumerate(chapt):
    print('file {} / {}: chapt {}'.format(i + 1, n_trials, ci))
    data = read_hdf5(stim_path.format(exp, nar) + stim_root.format(nar, ci))
    stim = data[stimulus]
    if len(stim.shape) > 1:
        stim = stim[0]

    print('resampling')
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=8,
                       verbose=False)
    del stim, data
    gc.collect()
    time.sleep(s_sleep)

    # run model in parallel using joblib
    print('running an model')
    start_time = datetime.datetime.now()
    an_mod = np.array(Parallel(n_jobs=11)([
        delayed(doanmodel)(stim_up, cfi) for cfi in range(len(cf))]))
    an_mod = an_mod.reshape(len(cf), len(anf_types), an_mod.shape[-1])
    td = datetime.datetime.now() - start_time
    print('Time for anf model: {}s'.format(td))

    print('running cn and ic model')
    bs_mod, w1, w3, w5 = dobsmodel(an_mod)

    an_mod_all[i] = an_mod[:, 0]
    bs_mod_all[i] = bs_mod
    w1_mod_all[i] = w1
    w3_mod_all[i] = w3
    w5_mod_all[i] = w5

    del stim_up, an_mod, bs_mod, w1, w3, w5
    gc.collect()
    time.sleep(s_sleep)

write_hdf5(save_path.format(exp) + save_root_model.format(exp, stimulus, nar),
           dict(an_mod_all=an_mod_all,
                bs_mod_all=bs_mod_all,
                w1_mod_all=w1_mod_all,
                w3_mod_all=w3_mod_all,
                w5_mod_all=w5_mod_all,
                cf=cf,
                scale=scale,
                fs_in=fs_in,
                fs_out=fs_out,
                fs_up=fs_up,
                n_trials=n_trials,
                chapt_inds=chapt,
                stimulus=stimulus), overwrite=True)
del an_mod_all, w1_mod_all, w3_mod_all, w5_mod_all

# %% load the stimuli
# NOTE: did this in separate step than above for memory issues
print('== loading and saving stimuli ==')
stim_all = np.zeros((n_trials, len_stim * int(fs_in)))
pulse_inds_all = []
pulses_all = np.zeros((n_trials, n_bands_all, len_stim * int(fs_out)))
stim_p = np.zeros((n_trials, len_stim * int(fs_out)))
stim_n = np.zeros((n_trials, len_stim * int(fs_out)))


for i, ci in enumerate(chapt):
    print('file {} / {}: chapt {}'.format(i + 1, n_trials, ci))
    data = read_hdf5(stim_path.format(exp, nar) + stim_root.format(nar, ci))
    stim = data[stimulus]
    if exp == 'Pilot':
        pinds = data['pulse_inds']
        pinds += data['fake_pulse_inds']
        pulse_inds = [(pi * float(fs_out) / fs_in).astype(int) for pi in
                      pinds[:n_bands_all]]
        del pinds
    else:
        pulse_inds = [(pi * float(fs_out) / fs_in).astype(int) for pi in
                      data['pulse_inds'][:n_bands_all]]
    pulses = np.zeros((n_bands_all, len_stim * int(fs_out)))
    for bi in range(n_bands_all):
        pulses[bi, pulse_inds[bi]] = 1.

    stim_all[i] = stim
    pulse_inds_all += [pulse_inds]
    pulses_all[i] = pulses
    stim_p[i] = resample(np.maximum(0, stim), fs_out, fs_in, npad='auto',
                         n_jobs=n_jobs)
    stim_n[i] = resample(np.maximum(0, -stim), fs_out, fs_in, npad='auto',
                         n_jobs=n_jobs)

    del data, stim, pulse_inds, pulses

data = dict(stim=stim_all,
            stim_p=stim_p,
            stim_n=stim_n,
            pulse_inds=pulse_inds_all,
            pulses=pulses_all,
            chapt_inds=chapt,
            fs_stim=fs_in,
            fs_out=fs_out,
            n_trials=n_trials,
            stimulus=stimulus)
write_hdf5(save_path.format(exp) + save_root_stim.format(exp, stimulus, nar),
           data, overwrite=True)
del data
gc.collect()

# %% deconvolution
print('== deconvolving ==')
fft_bs_mod = fft(bs_mod_all)

fft_stim_p = fft(stim_p)
fft_stim_n = fft(stim_n)
w_rect = (ifft((np.conj(fft_stim_p) * fft_bs_mod).mean(0) /
               (np.conj(fft_stim_p) * fft_stim_p).mean(0)).real * 0.5 + 0.5 *
          ifft((np.conj(fft_stim_n) * fft_bs_mod).mean(0) /
               (np.conj(fft_stim_n) * fft_stim_n).mean(0)).real) * scale
del fft_stim_p, fft_stim_n

fft_pulses = fft(pulses_all[:, 0])
w_pulses = ifft((np.conj(fft_pulses) * fft_bs_mod).mean(0) /
                (np.conj(fft_pulses) * fft_pulses).mean(0)).real * scale
del fft_pulses
gc.collect()

if stimulus == 'x_play_band':
    del w_pulses
    fft_pulses_true = fft(pulses_all[:, :n_bands])
    w_bands = ifft(
        (np.conj(fft_pulses_true) * fft_bs_mod[..., np.newaxis, :]).mean(0) /
        (np.conj(fft_pulses_true) * fft_pulses_true).mean(
            -2, keepdims=True).mean(0)).real * scale
    del fft_pulses_true
    fft_pulses_fake = fft(pulses_all[:, n_bands:])
    w_common = ifft(
        (np.conj(fft_pulses_fake) * fft_bs_mod[..., np.newaxis, :]).mean(0) /
        (np.conj(fft_pulses_fake) * fft_pulses_fake).mean(
            -2, keepdims=True).mean(0)).real * scale
    del fft_pulses_fake
    w_subtr = w_bands - w_common.mean(-2, keepdims=True)
    w_sumband = w_subtr.sum(-2) + w_common.mean(-2)
    gc.collect()

# %% saving
print('== saving ==')
data = dict(n_trials=n_trials,
            scale=scale,
            cf=cf,
            fs_in=fs_in,
            fs_out=fs_out,
            w_rect=w_rect)
if stimulus == 'x_play_band':
    data['w_bands'] = w_bands
    data['w_common'] = w_common
    data['w_subtr'] = w_subtr
    data['w_sumband'] = w_sumband
else:
    data['w_pulses'] = w_pulses

write_hdf5(save_path.format(exp) + save_root_responses.format(
    exp, stimulus, nar), data, overwrite=True)
del data
gc.collect()
