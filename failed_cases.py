
import numpy as np


success_rates = np.load("valid_success_rates.npy")
low_success_rates_idx = np.where(success_rates < 0.5)[0]
low_success_rates = success_rates[low_success_rates_idx]
for i, rate in zip(low_success_rates_idx, low_success_rates):
    print(f"Low success rate at index {i}: {rate}")


# Low success rate at index 7: 0.26
# Low success rate at index 13: 0.36
# Low success rate at index 21: 0.34
# Low success rate at index 26: 0.36
# Low success rate at index 29: 0.46
# Low success rate at index 41: 0.24
# Low success rate at index 43: 0.46
# Low success rate at index 50: 0.22
# Low success rate at index 54: 0.44
# Low success rate at index 55: 0.14
# Low success rate at index 81: 0.3
# Low success rate at index 87: 0.1
# Low success rate at index 94: 0.06
# Low success rate at index 100: 0.24
# Low success rate at index 102: 0.32
# Low success rate at index 103: 0.44
# Low success rate at index 106: 0.16
# Low success rate at index 119: 0.26
# Low success rate at index 121: 0.08
# Low success rate at index 125: 0.24
# Low success rate at index 129: 0.0
# Low success rate at index 141: 0.2
# Low success rate at index 145: 0.36
# Low success rate at index 147: 0.3
# Low success rate at index 161: 0.26
# Low success rate at index 162: 0.2
# Low success rate at index 163: 0.16
# Low success rate at index 172: 0.32
# Low success rate at index 179: 0.06
# Low success rate at index 188: 0.16
# Low success rate at index 193: 0.34
# Low success rate at index 200: 0.42
# Low success rate at index 204: 0.18
# Low success rate at index 210: 0.3
# Low success rate at index 211: 0.04
# Low success rate at index 212: 0.26
# Low success rate at index 217: 0.18 +
# Low success rate at index 222: 0.44
# Low success rate at index 224: 0.0
# Low success rate at index 230: 0.3
# Low success rate at index 235: 0.38
# Low success rate at index 236: 0.46
# Low success rate at index 237: 0.16 +
# Low success rate at index 244: 0.34
# Low success rate at index 248: 0.22 +
# Low success rate at index 261: 0.46 +
