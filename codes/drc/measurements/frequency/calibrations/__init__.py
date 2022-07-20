from ..responses import FrequencyResponse, SMOOTHING_WINDOW_SIZE, TREBLE_SMOOTHING_WINDOW_SIZE, Response
from ... import Sampler



class Microphone(Response):
    GAIN_dB = 0.0
    NULL_STRINGS = ('"', 'Sensitivity =', 'Sens Factor =', 'AGain =', 'dB', 'SERNO:', 'EARS Serial ')


    def __init__(self, responses = None,
                 nchannels = 1, idx_channel = 0,
                 sensitivity_dBFS = None,
                 gain_dB = GAIN_dB,
                 columns = f'Freq(Hz) SPL(dB) Phase(degrees)',
                 field_sep = ','):

        self.nchannels = nchannels
        self.idx_channel = idx_channel
        self.sensitivity_dBFS = sensitivity_dBFS
        self.gain_dB = gain_dB

        super().__init__(responses = responses,
                         columns = columns,
                         field_sep = field_sep)


    @property
    def header(self):
        return f'"Sensitivity = {self.sensitivity_dBFS} dB, AGain = {self.gain_dB} dB"'


    @classmethod
    def _extract_params(cls, str_responses, line_sep):

        line = str_responses.strip().split(line_sep)[0]

        for null_str in cls.NULL_STRINGS:
            line = line.replace(null_str, '')

        return line.split(',')


    def _extract_info(self, str_responses, line_sep):
        # "Sensitivity = -43.5 dB, AGain = 18 dB"
        params = self._extract_params(str_responses, line_sep)

        self.sensitivity_dBFS = float(params[0])
        self.gain_dB = float(params[1])


    def loads(self, str_responses, line_sep = '\n', *args, **kwargs):
        self._extract_info(str_responses, line_sep)
        return super().loads(str_responses, line_sep = line_sep, *args, **kwargs)


    def get_frequency_response(self, input_device_idx = None, *args, **kwargs):
        return Sampler.get_frequency_response(input_device_idx = input_device_idx,
                                              nchannels = self.nchannels, idx_channel = self.idx_channel,
                                              sensitivity_dB = self.sensitivity_dBFS,
                                              gain_dB = self.gain_dB,
                                              calibration = self.frequency_response,
                                              *args, **kwargs)



class Calibrator:

    @classmethod
    def calibrate(cls, fr_measurement, fr_target,
                  window_size = SMOOTHING_WINDOW_SIZE, treble_window_size = TREBLE_SMOOTHING_WINDOW_SIZE):
        fr_measurement = fr_measurement.get_smoothed(window_size, treble_window_size)
        fr_target = fr_target.get_smoothed(window_size, treble_window_size)
        fr_target.interpolate(f = fr_measurement.frequency)

        fr_measurement.target = fr_target.raw
        fr_measurement.error = fr_measurement.raw - fr_measurement.target
        fr_measurement.smoothen_fractional_octave(window_size = window_size,
                                                  treble_window_size = treble_window_size)
        fr_calibration = FrequencyResponse('Calibration',
                                           fr_measurement.frequency,
                                           fr_measurement.error_smoothed)
        fr_calibration.center(frequency = 1000)
        fr_calibration.interpolate()

        return fr_measurement, fr_calibration
