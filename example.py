"""
This data set consists of EEG data from 9 subjects. The cue-based BCI paradigm
consisted of four different motor imagery tasks, namely the imag- ination of
movement of the left hand (class 1), right hand (class 2), both feet (class 3),
and tongue (class 4). Two sessions on different days were recorded for each
subject. Each session is comprised of 6 runs separated by short breaks.
One run consists of 48 trials (12 for each of the four possible classes),
yielding a total of 288 trials per session.

The subjects were sitting in a comfortable armchair in front of a computer
screen. At the beginning of a trial (t = 0 s), a fixation cross appeared on
the black screen. In addition, a short acoustic warning tone was presented.
After two seconds (t = 2 s), a cue in the form of an arrow pointing either to
the left, right, down or up (corresponding to one of the four classes left
hand, right hand, foot or tongue) appeared and stayed on the screen for 1.25 s.
This prompted the subjects to perform the desired motor imagery task.
No feedback was provided. The subjects were ask to carry out the motor imagery
task until the fixation cross disappeared from the screen at t = 6 s.

Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm) were
used to record the EEG; the montage is shown in Figure 3 left. All signals were
recorded monopolarly with the left mastoid serving as reference and the right
mastoid as ground. The signals were sampled with. 250 Hz and bandpass-filtered
between 0.5 Hz and 100 Hz. The sensitivity of the amplifier was set to 100 μV.
An additional 50 Hz notch filter was enabled to suppress line noise

Source:
http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html
"""
# %%
from pathlib import Path
import moabb
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from torch import fmax

# download full dataset
dataset = BNCI2014001()
dataset.download()

# %%
event_id = dataset.event_id
# {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4}

data = dataset.get_data()
# data = {'subject_id' :           # 1, 2, 3, 4, 5, 6, 7, 8, 9
#             {'session_id':       # 'session_T', 'session_E'
#                 {'run_id': raw}  # 'run_{X}', X = 0, 1, 2, 3, 4, 5
#             }
#         }

subject = 1
raw = data[subject]['session_T']['run_0']
# %%

paradigm = MotorImagery(n_classes=4, fmin=2, fmax=49, tmin=-0.2, tmax=None,
                        baseline=(-0.2, 0))

subjects = [1]
X, y, metadata = paradigm.get_data(
    dataset=dataset, subjects=subjects, return_epochs=True)

# %%
