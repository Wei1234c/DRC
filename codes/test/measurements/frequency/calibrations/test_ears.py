from matplotlib import pyplot as plt

from drc.measurements.frequency.calibrations.miniDSP import EARS


ears = EARS()

fn_l = 'L_RAW_8603591.txt'
fn_r = 'R_RAW_8603591.txt'

ears.ears['left'].load(file_name = fn_l)
ears.ears['right'].load(file_name = fn_r, field_sep = None, n_header_lines = 12, line_sep = '\n')

for ear in ears.ears.values():
    print('serial_no', ear.serial_no)
    print('gain_dB', ear.gain_dB)
    print('sensitivity_factor_dBFS', ear.sensitivity_factor_dBFS)
    print('sensitivity_dBFS', ear.sensitivity_dBFS)
    print()
    plt.plot(ear.fs, ear.amps)

file_name = 'ears.txt'
ears.ears['left'].dump(file_name)

ears.ears['left'].load(file_name)
print(ears.ears['left'].responses)

plt.xscale('log')
plt.show()
