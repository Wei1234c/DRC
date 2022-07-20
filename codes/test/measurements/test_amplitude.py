from drc.measurements import Amplitude


voltage = 2.0
value_FS = 0.5
sensitivity_dBV_at_one_pa = -46
sensitivity_dBFS_at_one_pa = -31.9
gain_dB = 18

# Vrms vs dBSPL =======================
# voltage = 10 ** (-46 / 20)
dBSPL = Amplitude.voltage_to_dBSPL(voltage, sensitivity_dBV_at_one_pa, gain_dB)
print('dBSPL', dBSPL)

vrms = Amplitude.voltage_from_dBSPL(dBSPL, sensitivity_dBV_at_one_pa, gain_dB)
print('vrms', vrms)

assert abs(vrms - voltage) < 1e-7

sensitivity_dBV = Amplitude.get_sensitivity_dBV_by_voltage(voltage, dBSPL, gain_dB)
print('sensitivity_dBV', sensitivity_dBV)
assert abs(sensitivity_dBV - sensitivity_dBV_at_one_pa) < 1e-7

# FS vs dBSPL =======================
dBSPL = Amplitude.FS_to_dBSPL(value_FS, sensitivity_dBFS_at_one_pa, gain_dB)
print('dBSPL', dBSPL)

v_fs = Amplitude.FS_from_dBSPL(dBSPL, sensitivity_dBFS_at_one_pa, gain_dB)
print('v_fs', v_fs)

assert abs(v_fs - value_FS) < 1e-7

sensitivity_dBFS = Amplitude.get_sensitivity_dBFS_by_FS(value_FS, dBSPL, gain_dB)
print('sensitivity_dBFS', sensitivity_dBFS)
assert abs(sensitivity_dBFS - sensitivity_dBFS_at_one_pa) < 1e-7

# dBFS vs dBSPL =======================
dBSPL = Amplitude.dBFS_to_dBSPL(Amplitude._to_dB(value_FS), sensitivity_dBFS_at_one_pa, gain_dB)
print('dBSPL', dBSPL)

dBFS = Amplitude.dBFS_from_dBSPL(dBSPL, sensitivity_dBFS_at_one_pa, gain_dB)
print('dBFS', dBFS)

assert abs(dBFS - Amplitude._to_dB(value_FS)) < 1e-7

sensitivity_dBFS = Amplitude.get_sensitivity_dBFS_by_dBFS(dBFS, dBSPL, gain_dB)
print('sensitivity_dBFS', sensitivity_dBFS)
assert abs(sensitivity_dBFS - sensitivity_dBFS_at_one_pa) < 1e-7

# dBV vs dBSPL =======================
dBSPL = Amplitude.dBV_to_dBSPL(Amplitude._to_dB(voltage), sensitivity_dBV_at_one_pa, gain_dB)
print('dBSPL', dBSPL)

dBV = Amplitude.dBV_from_dBSPL(dBSPL, sensitivity_dBV_at_one_pa, gain_dB)
print('dBV', dBV)

assert abs(dBV - Amplitude._to_dB(voltage)) < 1e-7

sensitivity_dBV = Amplitude.get_sensitivity_dBV_by_dBV(dBV, dBSPL, gain_dB)
print('sensitivity_dBV', sensitivity_dBV)
assert abs(sensitivity_dBV - sensitivity_dBV_at_one_pa) < 1e-7

# dBV vs dBFS ============================
dBFS = Amplitude.dBV_to_dBFS(Amplitude._to_dB(voltage), sensitivity_dBV_at_one_pa, sensitivity_dBFS_at_one_pa, gain_dB)
print('dBFS', dBFS)

dBV = Amplitude.dBFS_to_dBV(dBFS, sensitivity_dBFS_at_one_pa, sensitivity_dBV_at_one_pa, gain_dB)
print('dBV', dBV)

assert abs(dBV - Amplitude._to_dB(voltage)) < 1e-7

# voltage vs FS ============================
vfs = Amplitude.voltage_to_FS(voltage, sensitivity_dBV_at_one_pa, sensitivity_dBFS_at_one_pa, gain_dB)
print('vfs', vfs)

vrms = Amplitude.FS_to_voltage(vfs, sensitivity_dBFS_at_one_pa, sensitivity_dBV_at_one_pa, gain_dB)
print('vrms', vrms)

assert abs(vrms - voltage) < 1e-7
