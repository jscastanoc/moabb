#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:31:25 2019

@author: jscastanoc
"""
import numpy as np
import pandas as pd
import mne
from .motor_imagery import BaseMotorImagery

class BaseRegressionBandPower(BaseMotorImagery):
    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == 'regression_bp'):
            ret = False

        # check if dataset has required events
        if self.events:
            if not set(self.events) <= set(dataset.event_id.keys()):
                ret = False

        # we should verify list of channels, somehow
        return ret
    
    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}
    
    def datasets(self):
        return ['PostHoc Labelling']
    
    def scoring(self, estimator, X, y):
        y_est = estimator.predict(X)
        return np.corrcoef(y_est,y)[0,1]
    
    def process_raw(self, raw, dataset):
        X, _, metadata = super().process_raw(raw, dataset)
        labels = raw.continous_labels
        
        return X, labels, metadata
    
class RegressionSingleBand(BaseRegressionBandPower):
    def __init__(self, fmin=8, fmax=32, **kwargs):
        super().__init__(filters=([fmin, fmax]), **kwargs)

class RegressionFilterBank(BaseRegressionBandPower):
    def __init__(self, filters=([4, 8],[8, 12], [12, 16], [16, 20], [20, 24],
                                [24, 28]), **kwargs):
        super().__init__(filters=filters, **kwargs)

class BaseRegressionFrequencyDomain(BaseRegressionBandPower):
    def __init__(self, frange = [4, 25], nfft = 100, **kwargs):
        self.frange = frange
        self.nfft = nfft
        super().__init__(**kwargs)
        
    def process_raw(self, raw, dataset):
        
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        channels = () if self.channels is None else self.channels

        # picks channels
        picks = mne.pick_types(raw.info, eeg=True, stim=False,
                               include=channels)

        # get events id
        event_id = self.used_events(dataset)

        # pick events, based on event_id
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        # get interval
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]
            
        epochs = mne.Epochs(raw, events, event_id=event_id,
                                tmin=tmin-0.15, tmax=tmax+0.15, proj=False,
                                baseline=None, preload=True,
                                verbose=False, picks=picks)
        
        ix_accept_segm = [ix for ix, drop_reason in enumerate(epochs.drop_log) if drop_reason == []]
        
        X, _, freqs_selected = preprocess_data_for_fspoc(epochs, 
                                          [tmin, tmax],
                                          frange=self.frange,
                                          nfft=self.nfft,
                                          n_jobs=8,
                                          return_data=False)
        class CovarianceFrequency(object):
            freqs_selected=None
            def __init__(self, cov):
                self.cov = cov
        CovarianceFrequency.freqs_selected = freqs_selected        
        X = np.array([CovarianceFrequency(cov) for cov in X])
        
        labels = self.get_labels(raw)[ix_accept_segm]
        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata
    
    def get_labels(self, raw):
        return raw.continous_labels 
