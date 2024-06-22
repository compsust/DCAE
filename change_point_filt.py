"""
    Change point filter for NILM signals
    Alejandro Rodriguez-SFU
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_levels(signal, trigger, padding=0):
    """
        Function that gets the levels of the aggregate
        Parameters:
            signal - 1d data (aggregate), numpy array.
            trigger - list or numpy array containing the event locations
            padding - Number of samples to the right and to the left of the trigger locations (optional).
        Returns: filtered signal
    """
    result = np.zeros(len(signal))
    for i in range(len(trigger)-1):
        inf_lim = trigger[i] + padding
        sup_lim = trigger[i+1] - padding
        # for the last position
        if i == len(trigger)-2:
            level = np.mean(signal[sup_lim:])
            result[trigger[i+1]:] = level
        # padding issue, maybe there's a better way to solve this
        if inf_lim > sup_lim:
            result[trigger[i]:trigger[i+1]] = result[trigger[i-1]]
        else:
            # just take the mean between regions
            level = np.mean(signal[inf_lim:sup_lim])
            result[trigger[i]:trigger[i+1]] = level
    return result


def cp_filter(signal, ws=60, n_std=5, debug=False):
    """
        Main function
        Parameters:
            signal - 1d data (aggregate), numpy array or pandas dataframe
            window_size - the window size for the rolling mean and standard deviation
            n_std - the number of standard deviations that will activate the filter
        Returns: filtered signal
    """
    # convert numpy signal to pandas
    signal =  pd.DataFrame(signal)

    noise_thres = 30 # noise

    # preprocessing
    noise_w = (signal[0] < noise_thres )
    signal.loc[noise_w] = 0

    # processing
    rolling_mean = signal.rolling(window=ws).agg("mean") # take the rolling mean of the aggregate
    rolling_std = signal.rolling(window=ws).agg("std") # take the rolling std of the aggregate
    upper_limit = rolling_mean + n_std*rolling_std
    lower_limit = rolling_mean - n_std*rolling_std
    trigger = abs(rolling_mean - signal) > n_std*rolling_std # check for the samples that exceed the +- bounds (in termns of standad deviations)

    # There will be some off events that won't get caught due to the the high variance
    # from the rolling stats performed forward.
    # Now we just have to calculate the rolling stats backwards
    rolling_mean_back = signal.iloc[::-1].rolling(window=ws).agg("mean").iloc[::-1]
    rolling_std_back = signal.iloc[::-1].rolling(window=ws).agg("std").iloc[::-1]
    upper_limit_back = rolling_mean_back + n_std*rolling_std_back
    lower_limit_back = rolling_mean_back - n_std*rolling_std_back
    trigger_back = abs(rolling_mean_back - signal) > n_std*rolling_std_back # in order to be able to compare in terms of std

    # getting the final trigger positions
    full_trigger = pd.DataFrame(np.max(np.hstack([trigger,trigger_back]),axis=1)) # union of the events, if forward or backwards captures something, it will be reflected here.
    # why doing the difference?
    edges = full_trigger.diff() # take the difference to find the edges (it will detect only True-False/False-True events). Works when, they're pretty close to each other
    edges.fillna(False, inplace=True)
    edges_pos = edges.loc[edges[0]].index.to_numpy() # returns the index of the positions where there's an edge
    """
        take mid-point of the events that are very close, take the mean of these points. That will be the position of the events
        that are pretty close to each other, then I just iterate through that list and grab the ones were actual things happening, that's why we get rid of the ones in the middle
        the reason why we don't have more than two peaks in the difference is because we only detect true-false, false-true. In this way, we prevent from getting something like [712,713,714]
    """
    final_trigger = ((edges_pos[:-1]+edges_pos[1:])/2).astype(int)[::2]

    # get the levels for our filter
    signal_filt = get_levels(signal, final_trigger, 0)
    # debug
    if debug:
        # get the positions where samples exceed the upper/lower limits
        forward_triggers = np.array(trigger[0].values.astype(int))
        backward_triggers = np.array(trigger_back[0].values.astype(int))
        trigger_loc = np.where(forward_triggers > 0)[0]
        trigger_back_loc = np.where(backward_triggers > 0)[0]

        plt.figure()
        plt.title('Forward')
        plt.plot(signal, label='Raw agg.')
        plt.plot(upper_limit, label='$\mu_t + n \sigma_t$')
        plt.plot(lower_limit, label='$\mu_t - n \sigma_t$')
        plt.plot(trigger_loc, signal[0].values[trigger_loc], '*k')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()

        plt.figure()
        plt.title('Backwards')
        plt.plot(signal, label='Raw agg.')
        plt.plot(upper_limit_back, label='$\mu_t + n \sigma_t$')
        plt.plot(lower_limit_back, label='$\mu_t - n \sigma_t$')
        plt.plot(trigger_back_loc, signal[0].values[trigger_back_loc], '*k')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()

        plt.figure()
        plt.plot(signal)
        plt.plot(trigger_loc, signal[0].values[trigger_loc], '*r', label='fwd')
        plt.plot(trigger_back_loc, signal[0].values[trigger_back_loc], '*k', label='bwd')
        plt.plot(full_trigger*1000, label='$(F \cup B)$')
        plt.plot(edges*1000,label='difference')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()
    #return signal_filt, final_trigger, edges_pos
    return signal_filt
# this does not make sense with the negative peaks
def cp_resample(x_filt, triggers, debug=False):
    delta_x = np.diff(x_filt)

    # calculate the number of samples between events
    event_t = np.concatenate([triggers[1:]-triggers[:-1], [0]])

    x_resampled = []

    # resample the signal
    for t, trigger in zip(event_t, triggers):
        x_resampled.append(np.tile(delta_x[trigger-1], int(t/10))) # repeat the delta values dependent on the duration between events

    x_resampled = np.concatenate(x_resampled)

    if debug:
        plt.figure()
        plt.hist(np.abs(x_resampled), bins=100, range=[-500,500]) # need to work on this still

        plt.figure()
        plt.scatter(np.abs(x_resampled), np.zeros(len(x_resampled)))
    return x_resampled
