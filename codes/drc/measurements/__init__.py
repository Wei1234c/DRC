from ..measurements.frequency.responses import FrequencyResponse, SMOOTHING_WINDOW_SIZE, TREBLE_SMOOTHING_WINDOW_SIZE
from ..sound import np, Channel, InputDevice



# https://www.hometheatershack.com/threads/understanding-spl-offset-umik-1.134857/


class Amplitude:

    PA_AT_0_DB_SPL = 20 / 1e6
    DB_SPL_AT_ONE_PA = 20 * np.log10(1 / PA_AT_0_DB_SPL)  # 94


    @staticmethod
    def _to_dB(value):
        return 20 * np.log10(value)


    @staticmethod
    def _from_dB(dB):
        return 10 ** (dB / 20)


    @classmethod
    def to_dBSPL(cls, value, sensitivity_dB, gain_dB, value_ref = 1):
        value_at_one_pa = cls._from_dB(sensitivity_dB) * value_ref
        dB_to_one_pa = cls._to_dB(value / value_at_one_pa)
        dBSPL = dB_to_one_pa - gain_dB + cls.DB_SPL_AT_ONE_PA

        return dBSPL


    @classmethod
    def from_dBSPL(cls, dBSPL, sensitivity_dB, gain_dB, value_ref = 1):
        value_at_one_pa = cls._from_dB(sensitivity_dB) * value_ref
        dB_to_one_pa = dBSPL - cls.DB_SPL_AT_ONE_PA + gain_dB

        return value_at_one_pa * cls._from_dB(dB_to_one_pa)


    # voltage vs dBSPL ============================

    @classmethod
    def voltage_to_dBSPL(cls, voltage, sensitivity_dBV_at_one_pa, gain_dB, value_ref = 1):
        return cls.to_dBSPL(voltage, sensitivity_dBV_at_one_pa, gain_dB, value_ref)


    @classmethod
    def voltage_from_dBSPL(cls, dBSPL, sensitivity_dBV_at_one_pa, gain_dB, value_ref = 1):
        return cls.from_dBSPL(dBSPL, sensitivity_dBV_at_one_pa, gain_dB, value_ref)


    # FS vs dBSPL ============================
    @classmethod
    def FS_to_dBSPL(cls, value_FS, sensitivity_dBFS_at_one_pa, gain_dB):
        return cls.to_dBSPL(value_FS, sensitivity_dBFS_at_one_pa, gain_dB)


    @classmethod
    def FS_from_dBSPL(cls, dBSPL, sensitivity_dBFS_at_one_pa, gain_dB):
        return cls.from_dBSPL(dBSPL, sensitivity_dBFS_at_one_pa, gain_dB)


    # dBV vs dBSPL ============================
    @classmethod
    def dBV_to_dBSPL(cls, dBV, sensitivity_dBV_at_one_pa, gain_dB, value_ref = 1):
        return cls.voltage_to_dBSPL(cls._from_dB(dBV), sensitivity_dBV_at_one_pa, gain_dB, value_ref)


    @classmethod
    def dBV_from_dBSPL(cls, dBSPL, sensitivity_dBV_at_one_pa, gain_dB, value_ref = 1):
        return cls._to_dB(cls.voltage_from_dBSPL(dBSPL, sensitivity_dBV_at_one_pa, gain_dB, value_ref))


    # dBFS vs dBSPL ============================
    @classmethod
    def dBFS_to_dBSPL(cls, dBFS, sensitivity_dBFS_at_one_pa, gain_dB):
        return cls.FS_to_dBSPL(cls._from_dB(dBFS), sensitivity_dBFS_at_one_pa, gain_dB)


    @classmethod
    def dBFS_from_dBSPL(cls, dBSPL, sensitivity_dBFS_at_one_pa, gain_dB):
        return cls._to_dB(cls.FS_from_dBSPL(dBSPL, sensitivity_dBFS_at_one_pa, gain_dB))


    # dBV vs dBFS ============================
    @classmethod
    def dBFS_to_dBV(cls, dBFS, sensitivity_dBFS_at_one_pa, sensitivity_dBV_at_one_pa, gain_dB, value_ref = 1):
        dBSPL = cls.dBFS_to_dBSPL(dBFS, sensitivity_dBFS_at_one_pa, gain_dB)
        return cls.dBV_from_dBSPL(dBSPL, sensitivity_dBV_at_one_pa, gain_dB, value_ref)


    @classmethod
    def dBV_to_dBFS(cls, dBV, sensitivity_dBV_at_one_pa, sensitivity_dBFS_at_one_pa, gain_dB, value_ref = 1):
        dBSPL = cls.dBV_to_dBSPL(dBV, sensitivity_dBV_at_one_pa, gain_dB, value_ref)
        return cls.dBFS_from_dBSPL(dBSPL, sensitivity_dBFS_at_one_pa, gain_dB)


    # voltage vs FS ============================
    @classmethod
    def voltage_to_FS(cls, voltage, sensitivity_dBV_at_one_pa, sensitivity_dBFS_at_one_pa, gain_dB, value_ref = 1):
        dBSPL = cls.voltage_to_dBSPL(voltage, sensitivity_dBV_at_one_pa, gain_dB, value_ref)
        return cls.FS_from_dBSPL(dBSPL, sensitivity_dBFS_at_one_pa, gain_dB)


    @classmethod
    def FS_to_voltage(cls, value_FS, sensitivity_dBFS_at_one_pa, sensitivity_dBV_at_one_pa, gain_dB, value_ref = 1):
        dBSPL = cls.FS_to_dBSPL(value_FS, sensitivity_dBFS_at_one_pa, gain_dB)
        return cls.voltage_from_dBSPL(dBSPL, sensitivity_dBV_at_one_pa, gain_dB, value_ref)


    # get sensitivity ============================
    @classmethod
    def _get_sensitivity(cls, value, dBSPL, gain_dB, value_ref = 1):
        dB_to_one_pa = dBSPL - cls.DB_SPL_AT_ONE_PA + gain_dB
        value_at_one_pa = value / cls._from_dB(dB_to_one_pa)

        return cls._to_dB(value_at_one_pa / value_ref)


    @classmethod
    def get_sensitivity_dBFS_by_FS(cls, value_FS, dBSPL, gain_dB):
        return cls._get_sensitivity(value_FS, dBSPL, gain_dB)


    @classmethod
    def calibrate_sensitivity_dBFS_by_FS(cls, value_FS, gain_dB, dBSPL = 94):
        return cls.get_sensitivity_dBFS_by_FS(value_FS, dBSPL, gain_dB)


    @classmethod
    def get_sensitivity_dBFS_by_dBFS(cls, dBFS, dBSPL, gain_dB):
        return cls._get_sensitivity(cls._from_dB(dBFS), dBSPL, gain_dB)


    @classmethod
    def get_sensitivity_dBV_by_voltage(cls, voltage, dBSPL, gain_dB, value_ref = 1):
        return cls._get_sensitivity(voltage, dBSPL, gain_dB, value_ref)


    @classmethod
    def get_sensitivity_dBV_by_dBV(cls, dBV, dBSPL, gain_dB, value_ref = 1):
        return cls._get_sensitivity(cls._from_dB(dBV), dBSPL, gain_dB, value_ref)


    # get gain ============================

    @classmethod
    def _get_gain_dB(cls, value, sensitivity, dBSPL, value_ref = 1):
        return cls._to_dB(value / value_ref) - dBSPL + cls.DB_SPL_AT_ONE_PA - sensitivity


    @classmethod
    def get_gain_dB_by_FS(cls, value_FS, sensitivity_dBFS, dBSPL):
        return cls._get_gain_dB(value_FS, sensitivity_dBFS, dBSPL)


    @classmethod
    def calibrate_gain_dB_by_FS(cls, value_FS, sensitivity_dBFS, dBSPL = 94):
        return cls.get_gain_dB_by_FS(value_FS, sensitivity_dBFS, dBSPL)


    @classmethod
    def get_gain_dB_by_dBFS(cls, dBFS, sensitivity_dBFS, dBSPL):
        return cls._get_gain_dB(cls._from_dB(dBFS), sensitivity_dBFS, dBSPL)


    @classmethod
    def get_gain_dB_by_voltage(cls, voltage, sensitivity_dBV, dBSPL, value_ref = 1):
        return cls._get_gain_dB(voltage, sensitivity_dBV, dBSPL, value_ref)


    @classmethod
    def get_gain_dB_by_dBV(cls, dBV, sensitivity_dBV, dBSPL, value_ref = 1):
        return cls._get_gain_dB(cls._from_dB(dBV), sensitivity_dBV, dBSPL, value_ref)



class Sampler:

    DEFAULT_FS = 48000
    SAMPLE_WIDTH = 4
    CHUNK_SIZE = 1024 * 16
    FREQ_LIMITS = (20, 20000)
    MEAN_SPL_FREQ_LIMITS = (200, 6000)
    MEAN_SPL_N_SAMPLINGS = 5
    DEFAULT_N_SAMPLINGS = 10

    SENSITIVITY_dB = -42.0
    GAIN_dB = 18.0


    @staticmethod
    def _get_samples(input_device_idx = None, chunk_size = CHUNK_SIZE,
                     nchannels = 1, idx_channel = 0,
                     framerate = DEFAULT_FS, sample_width = SAMPLE_WIDTH, n_samplings = 1):
        channels = []

        with InputDevice(device_index = input_device_idx, nchannels = nchannels, chunk_size = chunk_size,
                         framerate = framerate, sample_width = sample_width) as mic:

            for _ in range(n_samplings):
                data = mic.read_chunk()
                channels.append(Channel(data[idx_channel], framerate = framerate))

        return channels


    @classmethod
    def probe(cls, input_device_idx = None, framerate = DEFAULT_FS,
              nchannels = 1, idx_channel = 0,
              freq_lims = FREQ_LIMITS,
              sensitivity_dB = SENSITIVITY_dB, gain_dB = GAIN_dB,
              n_samplings = 1,
              *args, **kwargs):

        channels = cls._get_samples(input_device_idx = input_device_idx,
                                    nchannels = nchannels, idx_channel = idx_channel,
                                    framerate = framerate, n_samplings = n_samplings,
                                    *args, **kwargs)

        spectrums = tuple(ch.make_spectrum(full = False) for ch in channels)
        amps = np.mean(np.stack([sp.amps for sp in spectrums]), axis = 0)
        dBSPLs = Amplitude.to_dBSPL(amps / len(amps), sensitivity_dB, gain_dB)

        fs = spectrums[0].fs
        idx = np.where((fs >= min(freq_lims)) & (fs <= max(freq_lims)))

        return fs[idx], dBSPLs[idx], channels


    @classmethod
    def get_frequency_response(cls, input_device_idx = None, framerate = DEFAULT_FS,
                               nchannels = 1, idx_channel = 0,
                               freq_lims = FREQ_LIMITS,
                               sensitivity_dB = SENSITIVITY_dB, gain_dB = GAIN_dB,
                               window_size = SMOOTHING_WINDOW_SIZE,
                               treble_window_size = TREBLE_SMOOTHING_WINDOW_SIZE,
                               calibration = None, n_samplings = 1,
                               *args, **kwargs):

        fs, amps, _ = cls.probe(input_device_idx = input_device_idx, framerate = framerate,
                                nchannels = nchannels, idx_channel = idx_channel,
                                freq_lims = freq_lims,
                                sensitivity_dB = sensitivity_dB, gain_dB = gain_dB,
                                n_samplings = n_samplings,
                                *args, **kwargs)

        fr_sample = FrequencyResponse('temp', fs, amps)
        fr_smoothed = fr_sample.get_smoothed(window_size, treble_window_size)

        if calibration is not None:
            calibration.interpolate(f = fr_smoothed.frequency)
            fr_sample.raw -= calibration.raw
            fr_sample.smoothed -= calibration.raw
            fr_smoothed.raw -= calibration.raw

        return fr_sample, fr_smoothed


    @classmethod
    def get_spl(cls, input_device_idx, freq_lims = MEAN_SPL_FREQ_LIMITS, n_samplings = MEAN_SPL_N_SAMPLINGS,
                *args, **kwargs):
        _, fr_smoothed = cls.get_frequency_response(input_device_idx = input_device_idx,
                                                    freq_lims = freq_lims,
                                                    n_samplings = n_samplings,
                                                    *args, **kwargs)
        return fr_smoothed.raw.mean()
