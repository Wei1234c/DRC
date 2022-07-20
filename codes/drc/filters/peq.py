import numpy as np

from autoeq import biquad



class PEQs:
    SAMPLING_FREQUENCY = 48000
    DEFAULT_Q = np.sqrt(2) / 2
    LABELS = ('0B', '1B', '2B', '1A', '2A')
    FILTER_TYPES = {'PK' : 'peaking',
                    'LS' : 'low_shelf',
                    'LSC': 'low_shelf',
                    'HS' : 'high_shelf',
                    'HSC': 'high_shelf'}


    def __init__(self, param_sets = None, header = None, columns = None, field_sep = None):
        self.param_sets = [] if param_sets is None else param_sets
        self.header = header or 'Filter Settings file'
        self.columns = columns or f'Filter  No 0/1 TYPE     Fc_Hz          Gain_dB        Q'
        self.field_sep = field_sep or ' '


    @property
    def param_sets(self):
        return self._param_sets


    @param_sets.setter
    def param_sets(self, param_sets):
        self._param_sets = param_sets


    def add_peq(self, freq_Hz, Q = DEFAULT_Q, gain_dB = 0, filter_type = 'PK'):
        assert filter_type in self.FILTER_TYPES.keys()

        self._param_sets.append({'type'   : filter_type,
                                 'freq_Hz': freq_Hz,
                                 'gain_dB': gain_dB,
                                 'Q'      : Q})


    @staticmethod
    def from_freq_Q_gain_pairs(freq_Q_gain_pairs):
        peqs = PEQs()

        for (freq, Q, gain_dB) in freq_Q_gain_pairs:
            peqs.add_peq(freq_Hz = freq, Q = Q, gain_dB = gain_dB, filter_type = 'PK')

        return peqs


    @property
    def fs(self):
        return tuple(ps['freq_Hz'] for ps in self.param_sets)


    @classmethod
    def _label_coeffs(cls, coeffs, idx = None):
        labels = cls.LABELS

        if idx is not None:
            labels = [f'{e}{idx}' for e in labels]

        return list(zip(labels, coeffs))


    @classmethod
    def get_coefficients(cls, freq_Hz, Q = DEFAULT_Q, gain_dB = 0, idx = None, fs = SAMPLING_FREQUENCY,
                         filter_type = 'peaking', reset = False):
        if reset:
            a1, a2, b0, b1, b2 = 0.0, 0.0, 1.0, 0.0, 0.0
        else:
            func = getattr(biquad, filter_type)
            _, a1, a2, b0, b1, b2 = func(fc = freq_Hz, Q = Q, gain = gain_dB, fs = fs)

        return cls._label_coeffs((b0, b1, b2, a1, a2), idx)


    @property
    def max_gain_dB(self):
        return max(ps['gain_dB'] for ps in self.param_sets)


    def get_coefficient_sets(self, n_filters = None, fs = SAMPLING_FREQUENCY, reset = False):
        coeffs = []
        n_settings = len(self.param_sets)
        n_filters = n_settings if n_filters is None else n_filters
        assert n_filters >= n_settings

        for i in range(n_settings):
            param_set = self.param_sets[i]
            coeffs.extend(self.get_coefficients(idx = i + 1,
                                                freq_Hz = param_set['freq_Hz'],
                                                Q = param_set['Q'],
                                                gain_dB = param_set['gain_dB'],
                                                fs = fs,
                                                filter_type = self.FILTER_TYPES[param_set['type']],
                                                reset = reset))
        for i in range(n_settings, n_filters):
            coeffs.extend(self.get_coefficients(idx = i + 1,
                                                freq_Hz = None,
                                                reset = True))
        return coeffs


    def get_coefficient_sets_values(self, *arg, **kwargs):
        return tuple(coeff[1] for coeff in self.get_coefficient_sets(*arg, **kwargs))


    def dump(self, file_name, encoding = 'utf8'):
        lines = []

        for i in range(len(self.param_sets)):
            lines.append(
                f"Filter  {i + 1}: ON  {self.param_sets[i]['type']}       Fc   {self.param_sets[i]['freq_Hz']:.1f} Hz  Gain  {self.param_sets[i]['gain_dB']:.3f} dB  Q  {self.param_sets[i]['Q']:.3f}")

        lines = [self.header, self.columns] + lines

        with open(file_name, 'wt', encoding = encoding) as f:
            f.writelines('\n'.join(lines))


    def loads(self, str_settings, field_sep = ' ', n_header_lines = 9, line_sep = '\n'):
        field_sep = field_sep or self.field_sep
        str_settings = str_settings.strip()

        if field_sep == ' ':
            for _ in range(5):
                str_settings = str_settings.replace('  ', ' ')

        lines = str_settings.split(line_sep)[n_header_lines:]
        lines = [line[line.find('ON') + 2:] for line in lines if 'ON' in line]
        lines = [line.strip().split(field_sep) for line in lines]
        self._param_sets = []

        for i in range(len(lines)):
            if len(lines[i]) >= 7:
                self._param_sets.append({'type'   : lines[i][0].strip(),
                                         'freq_Hz': eval(lines[i][2]),
                                         'gain_dB': eval(lines[i][5]),
                                         'Q'      : eval(lines[i][8]) if len(lines[i]) == 9 and lines[i][7] == 'Q'
                                         else np.sqrt(2) / 2})


    def load(self, file_name, field_sep = None, n_header_lines = 2, line_sep = '\n', encoding = 'utf8'):
        field_sep = field_sep or self.field_sep

        with open(file_name, 'rt', encoding = encoding) as f:
            lines = f.readlines()

        self.loads(''.join(lines), field_sep = field_sep, n_header_lines = n_header_lines, line_sep = line_sep)
