from autoeq.constants import DEFAULT_BASS_BOOST_GAIN, DEFAULT_BASS_BOOST_FC, DEFAULT_BASS_BOOST_Q
from ...filters.peq import PEQs, np
from ...measurements.frequency import Octave



class ResponseEqualizer:

    DEFAULT_FS = 48000
    MAX_FILTERS = 10
    MAX_GAIN_dB = 12.0
    DEFAULT_Q = np.sqrt(2)


    @staticmethod
    def get_peq_filters(measurement,
                        compensation,
                        max_filters = MAX_FILTERS,
                        max_gain_dB = MAX_GAIN_dB,
                        bass_boost_gain = DEFAULT_BASS_BOOST_GAIN,
                        bass_boost_fc = DEFAULT_BASS_BOOST_FC,
                        bass_boost_q = DEFAULT_BASS_BOOST_Q,
                        fs = DEFAULT_FS,
                        *args, **kwargs):

        compensation.interpolate()

        peq_filters, n_peq_filters, peq_max_gains, fbeq_filters, n_fbeq_filters, fbeq_max_gain = \
            measurement.process(compensation = compensation,
                                max_filters = max_filters,
                                max_gain = max_gain_dB,
                                bass_boost_gain = bass_boost_gain,
                                bass_boost_fc = bass_boost_fc,
                                bass_boost_q = bass_boost_q,
                                equalize = True,
                                parametric_eq = True,
                                fs = fs,
                                *args, **kwargs)

        peqs = PEQs.from_freq_Q_gain_pairs(peq_filters)

        return measurement, peqs, n_peq_filters, peq_max_gains


    @staticmethod
    def get_fixed_band_filters(measurement,
                               compensation,
                               fc = Octave.FC_10_BANDs,
                               q = None,
                               max_gain_dB = MAX_GAIN_dB,
                               bass_boost_gain = DEFAULT_BASS_BOOST_GAIN,
                               bass_boost_fc = DEFAULT_BASS_BOOST_FC,
                               bass_boost_q = DEFAULT_BASS_BOOST_Q,
                               fs = DEFAULT_FS,
                               *args, **kwargs):

        compensation.interpolate()

        if q is None:
            _bands = Octave.get_octave_by_bands(len(fc), min(fc), max(fc))
            q = _bands[:, 3]

        peq_filters, n_peq_filters, peq_max_gains, fbeq_filters, n_fbeq_filters, fbeq_max_gain = \
            measurement.process(compensation = compensation,
                                fc = fc,
                                q = q,
                                ten_band_eq = False,
                                max_gain = max_gain_dB,
                                bass_boost_gain = bass_boost_gain,
                                bass_boost_fc = bass_boost_fc,
                                bass_boost_q = bass_boost_q,
                                equalize = True,
                                fixed_band_eq = True,
                                fs = fs,
                                *args, **kwargs)

        peqs = PEQs.from_freq_Q_gain_pairs(fbeq_filters)

        return measurement, peqs, n_fbeq_filters, fbeq_max_gain


    @classmethod
    def get_eqapo_graphic_eq(cls, measurement, compensation,
                             file_path = None, fc = Octave.FC_31_BANDs, normalize = True,
                             *auto_eq_args, **auto_eq_kwargs):

        measurement, _, _, _ = cls.get_peq_filters(measurement, compensation, *auto_eq_args, **auto_eq_kwargs)
        return measurement.get_eqapo_graphic_eq(file_path, fc, normalize)
