from ..responses import Response
from ....measurements import Amplitude
from ....measurements.frequency.calibrations import Microphone



class UMIK1(Microphone):
    GAIN_dB = 18.0
    dBSPL_of_SENSITIVITY_FACTOR = 100.0
    SENSITIVITY_FACTOR_dBFS = -0.667
    PREAMP_dB = 24.0


    def __init__(self, responses = None,
                 nchannels = 1, idx_channel = 0,
                 sensitivity_factor_dBFS = None,
                 gain_dB = GAIN_dB,
                 columns = f'Freq(Hz) SPL(dB) Phase(degrees)',
                 field_sep = '\t'):
        self.nchannels = nchannels
        self.idx_channel = idx_channel
        self.sensitivity_factor_dBFS = sensitivity_factor_dBFS or self.SENSITIVITY_FACTOR_dBFS
        self.gain_dB = gain_dB
        self.serial_no = None

        Response.__init__(self, responses = responses, columns = columns, field_sep = field_sep)


    @property
    def header(self):
        return f'"Sens Factor = {self.sensitivity_factor_dBFS} dB, AGain = {self.gain_dB} dB, SERNO: {self.serial_no}"'


    @classmethod
    def get_sensitivity_dBFS(cls, sensitivity_factor_dBFS, gain_dB = GAIN_dB, preamp_dB = PREAMP_dB):
        # https://www.hometheatershack.com/threads/understanding-spl-offset-umik-1.134857/
        return Amplitude.get_sensitivity_dBFS_by_dBFS(dBFS = sensitivity_factor_dBFS,
                                                      dBSPL = cls.dBSPL_of_SENSITIVITY_FACTOR,
                                                      gain_dB = gain_dB + preamp_dB)


    @property
    def sensitivity_dBFS(self):
        return self.get_sensitivity_dBFS(self.sensitivity_factor_dBFS, self.gain_dB)


    @sensitivity_dBFS.setter
    def sensitivity_dBFS(self, sensitivity_dBFS):
        self.sensitivity_factor_dBFS = self.get_sensitivity_factor_dBFS(sensitivity_dBFS, self.gain_dB)


    @classmethod
    def get_sensitivity_factor_dBFS(cls, sensitivity_dBFS, gain_dB = GAIN_dB):
        return Amplitude.dBFS_from_dBSPL(dBSPL = cls.dBSPL_of_SENSITIVITY_FACTOR,
                                         sensitivity_dBFS_at_one_pa = sensitivity_dBFS,
                                         gain_dB = gain_dB + cls.PREAMP_dB)


    def _extract_info(self, str_responses, line_sep):
        # "Sens Factor =-0.667dB, AGain =18dB, SERNO: 7103946"
        params = self._extract_params(str_responses, line_sep)

        self.sensitivity_factor_dBFS = float(params[0])
        self.gain_dB = float(params[1])
        self.serial_no = params[2].strip()



class EAR(UMIK1):
    N_HEADER_LINES = 12
    SENSITIVITY_FACTOR_dBFS = -0.8


    def __init__(self, field_sep = ' ', *args, **kwargs):
        super().__init__(field_sep = field_sep, *args, **kwargs)


    @property
    def header(self):
        return f'"Sens Factor = {self.sensitivity_factor_dBFS} dB, EARS Serial  {self.serial_no}"'


    def _extract_info(self, str_responses, line_sep):
        # "Sens Factor =-0.8dB, EARS Serial 860-3591, compensation RAW V1"
        params = self._extract_params(str_responses, line_sep)

        self.sensitivity_factor_dBFS = float(params[0])
        self.serial_no = params[1].strip()



class EARS:
    N_CHANNELS = 2


    def __init__(self):
        self.ears = {'left' : EAR(nchannels = self.N_CHANNELS, idx_channel = 0),
                     'right': EAR(nchannels = self.N_CHANNELS, idx_channel = 1)}


    def load(self, calibration_files):
        for channel in self.ears.keys():
            self.ears[channel].load(calibration_files[channel])
