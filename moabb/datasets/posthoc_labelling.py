#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:24:26 2019

@author: jscastanoc
"""
import os
from os.path import join
import numpy as np
import mne
from .base import  BaseDataset
from scipy.io import loadmat


DATA_POSTHOC = join(os.environ['PATH_DATA'],'posthoc_data')
MONTAGE_FILE = join(DATA_POSTHOC,'mnt_standard_1005_bsdlab.elc')
class PosthocLabelling(BaseDataset):
    
    
    def __init__(self, target_fband = 'delta', discrete_labels = False):
        self.target_fband = target_fband
        self.discrete_labels = discrete_labels
        learning_task = 'classification_bp' if discrete_labels else 'regression_bp'
        super().__init__( list(range(1,8)), 1, {'dummy':1},               
                             'PostHoc Labelling', [0,1], learning_task)
    
    def _get_single_subject_data(self, subject):
        eegdata_mat = loadmat(self.data_path(subject))
        
        ch_names = [clab[0] for clab in eegdata_mat['cnt'][0][0][0][0].tolist()]
        sfreq = eegdata_mat['cnt'][0][0][1][0][0]
        X = 1e-6*eegdata_mat['cnt'][0][0][-1].T
        iart = eegdata_mat['iart']
        
        mnt = mne.channels.read_montage(MONTAGE_FILE)
        info = mne.create_info(ch_names, sfreq, montage=mnt, ch_types='eeg')
        raw = mne.io.RawArray(X, info)
        
        
        vmrk_tstamp_sec = eegdata_mat['vmrk'][0][0][0]/1000
        vmrk_tstamp_sec = np.delete(vmrk_tstamp_sec,iart) 
        
        
        # build event array
        sample_ix = np.squeeze(np.ceil(120*(vmrk_tstamp_sec)).astype(int))[1:-1]
        events = np.zeros((sample_ix.shape[0],3)).astype(int)
        events[:,-1] = -1
        events[:,0] = sample_ix
        
        info_stim_ch = mne.create_info(['STI 014'], sfreq, ch_types='stim')
        stim_ch = mne.io.RawArray(np.zeros((1,raw.get_data().shape[1])), info_stim_ch)
        raw.add_channels([stim_ch])
        raw.add_events(events)
        
        labels = self._get_labels(subject, raw)
        if not self.discrete_labels:
            raw.continous_labels = labels
        else:
            events[:,-1] = labels
            raw.add_events(events, replace=True)
            
        
        
        data = {'session_0':
                    {'run_0': raw}
                }
        return data    
    
    def _get_labels(self, subject, raw, ix_comp = 'rand'):
        
        ica_decomp = loadmat(join(DATA_POSTHOC,'S%d.mat_ICA_decomp.mat' % subject))
        ica_decomp = ica_decomp[self.target_fband]        
        W_ica = ica_decomp[0][0][1]
        
        ev = mne.find_events(raw,shortest_event=0, verbose=False) 
        
        if ix_comp == 'rand':
            ix_comp = np.random.randint(0, W_ica.shape[1])
        picks = mne.pick_types(raw.info,eeg=True, stim=False)
        source = W_ica[:,ix_comp].T @ raw.get_data(picks=picks)        
        info_tsource = mne.create_info(['target_s'], raw.info['sfreq'], ch_types='eeg')
        raw_tsource = mne.io.RawArray(np.atleast_2d(source), info_tsource)
        epochs_tsource = mne.Epochs(raw_tsource, ev, 
                                    self.event_id, 
                                    tmin=self.interval[0], 
                                    tmax=self.interval[1])
        
        z = np.squeeze(np.log(np.mean(epochs_tsource.get_data()**2,axis=-1)))
        
        threshold = np.median(z)
        y = np.ones(z.shape)
        y[z > threshold] = 2
        #pdb.set_trace()
        if self.discrete_labels:
            return y.astype(int)
        else:
            return z        
        
    
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        return join(DATA_POSTHOC,'S%d.mat' % subject)