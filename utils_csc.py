import numpy as np
from tqdm import tqdm
import mne


def process_raw(raw, dataset, fmin=2, fmax=None, make_epochs=False,
                tmin=-2, tmax=None, baseline=None):
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

    if make_epochs:

        # get interval
        tmin = tmin + dataset.interval[0]
        if tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = tmax + dataset.interval[0]

        # epoch data
        if baseline is not None:
            baseline = (
                baseline[0] + dataset.interval[0],
                baseline[1] + dataset.interval[0],
            )
            bmin = baseline[0] if baseline[0] < tmin else tmin
            bmax = baseline[1] if baseline[1] > tmax else tmax
        else:
            bmin = tmin
            bmax = tmax
        epochs = mne.Epochs(
            raw_f,
            events,
            event_id=event_id,
            tmin=bmin,
            tmax=bmax,
            proj=False,
            baseline=baseline,
            preload=True,
            verbose=False,
            picks=picks,
            event_repeated="drop",
            on_missing="ignore",
        )
        if bmin < tmin or bmax > tmax:
            epochs.crop(tmin=tmin, tmax=tmax)
        X = dataset.unit_factor * epochs.get_data()

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])

        return X, labels

    else:
        return raw_f.get_data(picks=picks), events


def get_data(dataset, subjects=None, session='all', fmin=2, fmax=None,
             make_epochs=False, tmin=-2, tmax=None, baseline=None):
    """

    Parameters
    ----------
    subjects: List of int | None
        List of subject number
        if None, all subjects are taken into account

    session : str
        what session to take into account
        'all', 'session_T' or 'session_E'


    Returns
    -------

    """

    data = dataset.get_data(subjects)
    X = np.array([]) if make_epochs else []
    labels = []

    for subject, sessions in tqdm(data.items(), desc=f"pre-processing", disable=True if len(data) == 1 else False):
        for this_session, runs in sessions.items():
            if not (session == 'all' or this_session == session):
                continue

            for run, raw in runs.items():
                x, lbs = process_raw(
                    raw, dataset, fmin=fmin, fmax=fmax,
                    make_epochs=make_epochs, tmin=tmin, tmax=tmax,
                    baseline=baseline)

                if x is None:
                    # this mean the run did not contain any selected event
                    # go to next
                    continue

                if make_epochs:
                    X = np.append(X, x, axis=0) if len(X) else x
                    labels = np.append(labels, lbs, axis=0)
                else:
                    X.append(x)               

    if make_epochs:
        return np.array(X), labels
    else:
        return np.array(X)
