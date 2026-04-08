import numpy as np
import matplotlib.pyplot as plt

# Parameters from the numerical example
true_x = 10000.0  # True constant range (m)
R = 100.0         # Measurement noise variance (m^2)

# Initial estimate and uncertainty
x_hat = 10200.0   # Initial state estimate (m)
P = 400.0         # Initial uncertainty (m^2)

# Generate noisy measurements
# First measurement from the example, others simulated
num_measurements = 10
np.random.seed(42) # Set seed for reproducible random numbers
z = true_x + np.sqrt(R) * np.random.randn(num_measurements)
z[0] = 10050.0    # Set first measurement as per the example

# Initialize arrays to store estimates and uncertainties
estimates = np.zeros(num_measurements + 1)
uncertainties = np.zeros(num_measurements + 1)
estimates[0] = x_hat
uncertainties[0] = P

# Kalman filter loop
for k in range(num_measurements):
    # Prediction step (state is constant, no process noise)
    x_hat_minus = x_hat
    P_minus = P
    
    # Compute Kalman gain
    K = P_minus / (P_minus + R)
    
    # Update step
    x_hat = x_hat_minus + K * (z[k] - x_hat_minus)
    P = (1 - K) * P_minus
    
    # Store results
    estimates[k + 1] = x_hat
    uncertainties[k + 1] = P

# Display results
print("Step-by-step Estimates (m):")
print(np.round(estimates, 2))
print("\nStep-by-step Uncertainties (m^2):")
print(np.round(uncertainties, 2))

# Plot the results
plt.figure(figsize=(10, 8))

# Subplot 1: Estimates
plt.subplot(2, 1, 1)
plt.plot(range(num_measurements + 1), estimates, 'b-o', label='Estimate')
plt.axhline(y=true_x, color='r', linestyle='--', label='True Value')
plt.xlabel('Measurement Step')
plt.ylabel('Estimated Range (m)')
plt.title('Kalman Filter State Estimation')
plt.legend()
plt.grid(True)

# Subplot 2: Uncertainties
plt.subplot(2, 1, 2)
plt.plot(range(num_measurements + 1), np.sqrt(uncertainties), 'g-o')
plt.xlabel('Measurement Step')
plt.ylabel('Standard Deviation (m)')
plt.title('Uncertainty Reduction')
plt.grid(True)

plt.tight_layout()
plt.show()
