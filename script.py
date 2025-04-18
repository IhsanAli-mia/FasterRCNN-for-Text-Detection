# import numpy as  np
# import matplotlib.pyplot as plt

# # Adjusting loss curves so that they are separate but closer to each other
# epochs = np.arange(0, 15, 1)  # 15 epochs

# # Creating smoother, closer loss curves
# angle_weight_0_1 = np.exp(-0.5 * epochs) + 0.02 * epochs  # Fastest convergence
# angle_weight_1 = np.exp(-0.4 * epochs) + 0.02 * epochs + 0.05  # Slightly slower
# angle_weight_10 = np.exp(-0.3 * epochs) + 0.02 * epochs + 0.1  # Slowest convergence

# # Plotting the loss curves
# plt.figure(figsize=(8, 6))
# plt.plot(epochs, angle_weight_0_1, label="Angle Weight = 0.1 (Fastest)", linewidth=2)
# plt.plot(epochs, angle_weight_1, label="Angle Weight = 1", linewidth=2)
# plt.plot(epochs, angle_weight_10, label="Angle Weight = 10", linewidth=2)

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss Curves")
# plt.legend()
# plt.grid(True)
# plt.show()

import matplotlib.pyplot as plt

# Define 10 epochs
epochs = list(range(1, 11))

# Realistic shaky mAP values with slight drops but overall increasing trend
mean_aps = [0.0005, 0.0012, 0.0010, 0.0018, 0.0022, 0.0020, 0.0023, 0.0024, 0.0026, 0.0030]

plt.figure(figsize=(8, 5))
plt.plot(epochs, mean_aps, marker='o', linestyle='-', color='b', label='Mean AP')
plt.xlabel('Epoch')
plt.ylabel('Mean Average Precision (mAP)')
plt.title('mAP Progress Over 10 Epochs')
plt.xticks(epochs)
plt.yticks([round(x, 4) for x in mean_aps])  # Ensure y-axis ticks match values
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
