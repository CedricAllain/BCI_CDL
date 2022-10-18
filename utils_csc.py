import numpy as np
from tqdm import tqdm
import mne

def process_raw(raw, dataset, fmin=2, fmax=None):
    """

    """
    event_id = dataset.event_id

    stim_channels = mne.utils._get_stim_channel(
        None, raw.info, raise_error=False)

    if len(stim_channels) > 0:
        events = mne.find_events(raw, shortest_event=0, verbose=False)
    else:
        try:
            events, _ = mne.events_from_annotations(
                raw, event_id=event_id, verbose=False
            )
        except ValueError:
            print(f"No matching annotations in {raw.filenames}")
            return

    picks = mne.pick_types(raw.info, eeg=True, stim=False)
    events = mne.pick_events(events, include=list(event_id.values()))

    # filter data
    raw_f = raw.copy().filter(
        fmin, fmax, method="iir", picks=picks, verbose=False
    )

    return raw_f.get_data(picks=picks), events


def get_data(dataset, subjects=None, fmin=2, fmax=None):
    """

    Parameters
    ----------
    subjects: List of int | None
        List of subject number
        if None, all subjects are taken into account

    Returns
    -------

    """

    data = dataset.get_data(subjects)
    X = []

    for subject, sessions in tqdm(data.items(), desc=f"pre-processing", disable=True if len(data)==1 else False):
        for session, runs in sessions.items():
            for run, raw in runs.items():
                x, _ = process_raw(
                    raw, dataset, fmin=fmin, fmax=fmax)

                if x is None:
                    # this mean the run did not contain any selected event
                    # go to next
                    continue

                X.append(x)

    return np.array(X)