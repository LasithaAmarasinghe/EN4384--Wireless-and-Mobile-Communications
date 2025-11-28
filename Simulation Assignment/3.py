import numpy as np
import matplotlib.pyplot as plt

# Parameters
mT = 4  # Number of transmitter antennas
mR = 4  # Number of receiver antennas
noise_variance = 1  # Noise variance
transmitter_power = [1, 2, 5, 10]  # Transmitter power values (W)
num_realizations = 1000
capacity_results = np.zeros((4, len(transmitter_power), num_realizations))

for j, Pt in enumerate(transmitter_power):
    for i in range(num_realizations):
        # Generating random channel matrix H
        H = np.sqrt(1 / 2) * (np.random.randn(mR, mT) + 1j * np.random.randn(mR, mT))

        # Singular value decomposition of H
        singular_values = np.linalg.svd(H, compute_uv=False)
        eig_vals = singular_values**2 / noise_variance

        # Equal power allocation
        power_equal = Pt / mT
        capacity_results[0, j, i] = np.sum(np.log2(1 + (power_equal * eig_vals)))

        # Channel inversion power allocation
        power_channel_inv = Pt / np.sum(1 / eig_vals)
        capacity_results[1, j, i] = mT * np.log2(1 + power_channel_inv)

        # Allocating all power to the best eigenmode
        capacity_results[2, j, i] = np.log2(1 + (Pt * np.max(eig_vals)))

        # Waterfilling power allocation
        WF_pow = (Pt + np.sum(1 / eig_vals)) / len(eig_vals) - 1 / eig_vals
        while np.any(WF_pow < 0):
            non_positive_indices = WF_pow <= 0
            positive_indices = WF_pow > 0
            remaining_eigenvalues = eig_vals[positive_indices]
            num_positive = len(remaining_eigenvalues)
            WF_pow_temp = (Pt + np.sum(1 / remaining_eigenvalues)) / num_positive - 1 / remaining_eigenvalues
            WF_pow[non_positive_indices] = 0
            WF_pow[positive_indices] = WF_pow_temp

        capacity_results[3, j, i] = np.sum(np.log2(1 + (WF_pow * eig_vals)))

# Calculate average capacity
average_capacity = np.mean(capacity_results, axis=2)

# Plot Equal Power Allocation with crosses
plt.figure(figsize=(20, 10))
plt.plot(transmitter_power, average_capacity[0, :], 'xr-', label='Equal Power Allocation')  # Red with crosses
plt.xlabel('Total Tx Power (W)')
plt.ylabel('Channel Capacity (bits/s/Hz)')
plt.title('Equal Power Allocation')
plt.grid()

# Plot Channel Inversion Power Allocation with crosses
plt.figure(figsize=(20, 10))
plt.plot(transmitter_power, average_capacity[1, :], 'xg-', label='Channel Inversion Power Allocation')  # Green with crosses
plt.xlabel('Total Tx Power (W)')
plt.ylabel('Channel Capacity (bits/s/Hz)')
plt.title('Channel Inversion Power Allocation')
plt.grid()

# Plot Best Eigenmode Power Allocation with crosses
plt.figure(figsize=(20, 10))
plt.plot(transmitter_power, average_capacity[2, :], 'xb-', label='Best Eigenmode Power Allocation')  # Blue with crosses
plt.xlabel('Total Tx Power (W)')
plt.ylabel('Channel Capacity (bits/s/Hz)')
plt.title('Best Eigenmode Power Allocation')
plt.grid()

# Plot Waterfilling Power Allocation with crosses
plt.figure(figsize=(20, 10))
plt.plot(transmitter_power, average_capacity[3, :], 'xm-', label='Waterfilling Power Allocation')  # Magenta with crosses
plt.xlabel('Total Tx Power (W)')
plt.ylabel('Channel Capacity (bits/s/Hz)')
plt.title('Waterfilling Power Allocation')
plt.grid()

# Combine all plots into a single figure with updated markers (crosses)
plt.figure(figsize=(20, 10))
plt.plot(transmitter_power, average_capacity[0, :], 'xr-', label='Equal Power Allocation') 
plt.plot(transmitter_power, average_capacity[1, :], 'xg-', label='Channel Inversion Power Allocation')  
plt.plot(transmitter_power, average_capacity[2, :], 'xb-', label='Best Eigenmode Power Allocation')  
plt.plot(transmitter_power, average_capacity[3, :], 'xm-', label='Waterfilling Power Allocation')  
plt.xlabel('Total Tx Power (W)')
plt.ylabel('Channel Capacity (bits/s/Hz)')
plt.title('Average Achievable Rates of a 4x4 MIMO System for Different Power Allocation Schemes')
plt.legend()
plt.grid()
plt.show()

# Create a table to display the results
import pandas as pd

# Convert average_capacity to a DataFrame for easier table manipulation
table_data = np.vstack([average_capacity[0, :], 
                        average_capacity[1, :], 
                        average_capacity[2, :], 
                        average_capacity[3, :]]).T

# Create a DataFrame with columns for each power allocation scheme
df = pd.DataFrame(table_data, columns=[
    'Equal Power Allocation', 
    'Channel Inversion Power Allocation', 
    'Best Eigenmode Power Allocation', 
    'Waterfilling Power Allocation'], 
    index=[f'{pt} W' for pt in transmitter_power])

# Display the table
print(df)

