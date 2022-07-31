import copy

import numpy as np

from autoeq import frequency_response
from drc.measurements.frequency import plot_spectrum
from .. import Octave


SMOOTHING_WINDOW_SIZE = 1 / 12
TREBLE_SMOOTHING_WINDOW_SIZE = 1



class FrequencyResponse(frequency_response.FrequencyResponse):
    # from : https://github.com/jaakkopasanen/AutoEq

    FREQUENCY_TO_CENTER = [100, 8000]


    def __add__(self, other):
        other.interpolate(f = self.frequency)
        return FrequencyResponse('Sum of frequency responses', self.frequency, self.raw + other.raw)


    @classmethod
    def read_csv_from_url(cls, url, file_path = 'temp.csv'):
        import pandas as pd

        pd.read_csv(url).to_csv(file_path, index = False)
        return cls.read_from_csv(file_path)


    @property
    def responses(self):
        return tuple((f, amp) for f, amp in zip(self.frequency, self.raw))


    def get_smoothed(self, window_size = SMOOTHING_WINDOW_SIZE, treble_window_size = TREBLE_SMOOTHING_WINDOW_SIZE):
        if len(self.smoothed) == 0:
            self.smoothen_fractional_octave(window_size = window_size, treble_window_size = treble_window_size)

        return FrequencyResponse(self.name, self.frequency, self.smoothed)


    def center(self, frequency = None):
        frequency = self.FREQUENCY_TO_CENTER if frequency is None else frequency

        return super().center(frequency)


    def plot(self, *args, **kwargs):
        return self.plot_graph(*args, **kwargs)


    def get_eqapo_graphic_eq(self, file_path = None, fc = Octave.FC_31_BANDs, normalize = True):
        fr = self.__class__(name = 'temp', frequency = self.frequency, raw = self.equalization)
        fr.interpolate(f = fc)

        if normalize:
            fr.raw -= np.max(fr.raw) + frequency_response.PREAMP_HEADROOM
            if fr.raw[0] > 0.0:  # Prevent bass boost below lowest frequency
                fr.raw[0] = 0.0

        s = '\n'.join(['{f}, {a:.1f}'.format(f = f, a = a) for f, a in zip(fr.frequency, fr.raw)])

        if file_path is not None:
            with open(file_path, 'wt', encoding = 'utf-8') as f:
                f.write('Equalizer APO Graphic EQ: \n' + s)

        return s



class Response:
    # mostly for handling .csv, .txt files

    N_HEADER_LINES = 2


    def __init__(self, responses = None, columns = None, field_sep = None):
        self.responses = [] if responses is None else responses
        self.columns = columns or f'\tHz\tMag (dB)\tdeg'
        self.field_sep = field_sep or ','


    def copy(self):
        return copy.deepcopy(self)


    @property
    def responses(self):
        return np.array(self._responses)


    @responses.setter
    def responses(self, responses):
        self._responses = responses


    @property
    def fs(self):
        return self.responses[:, 0]


    @property
    def frequency(self):
        return self.fs


    @property
    def amps(self):
        return self.responses[:, 1]


    @property
    def frequency_response(self):
        if len(self.responses):
            return FrequencyResponse('Frequency Response', self.fs, self.amps)


    @property
    def raw(self):
        return self.amps


    @property
    def phases(self):
        return self.responses[:, 2]


    @property
    def header(self):
        return 'Frequency Response'


    def dump(self, file_name, field_sep = None):
        field_sep = field_sep or self.field_sep

        data = [field_sep.join([str(e) for e in r]) for r in self._responses]
        lines = [self.header, self.columns] + data

        with open(file_name, 'wt') as f:
            f.writelines('\n'.join(lines))


    def loads(self, str_responses, field_sep = None, n_header_lines = None, line_sep = '\n'):
        field_sep = field_sep or self.field_sep
        n_header_lines = n_header_lines or self.N_HEADER_LINES
        str_responses = str_responses.strip()

        if field_sep == ' ':
            for _ in range(5):
                str_responses = str_responses.replace('  ', ' ')

        lines = str_responses.split(line_sep)[n_header_lines:]
        self._responses = [line.strip().split(field_sep) for line in lines]

        for i in range(len(self._responses)):
            self._responses[i] = tuple(eval(e) for e in self._responses[i])

        return self


    def load(self, file_name, field_sep = None, n_header_lines = None, line_sep = '\n'):
        field_sep = field_sep or self.field_sep
        n_header_lines = n_header_lines or self.N_HEADER_LINES

        with open(file_name, 'rt') as f:
            lines = f.readlines()

        return self.loads(''.join(lines), field_sep = field_sep, n_header_lines = n_header_lines, line_sep = line_sep)


    def plot(self):
        plot_spectrum(self.fs, self.amps)
