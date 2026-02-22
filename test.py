import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

# Time parameters
sample_rate = 1000  # samples per second
duration = 2  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))

# Create a square wave pulse (centered)
pulse_width = 0.3  # seconds
square_pulse = np.zeros_like(t)
center = len(t) // 2
pulse_samples = int(pulse_width * sample_rate)
start = center - pulse_samples // 2
end = start + pulse_samples
square_pulse[start:end] = 1.0

# Create a sawtooth wave pulse (centered)
sawtooth_pulse = np.zeros_like(t)
saw_samples = int(pulse_width * sample_rate)
saw_start = center - saw_samples // 2
saw_end = saw_start + saw_samples
sawtooth_pulse[saw_start:saw_end] = np.linspace(0, 1, saw_samples)

# Calculate full cross-correlation and convolution
cross_corr = signal.correlate(square_pulse, sawtooth_pulse, mode='same')
cross_corr_norm = cross_corr / np.max(np.abs(cross_corr))

conv_result = signal.convolve(square_pulse, sawtooth_pulse, mode='same')
conv_result_norm = conv_result / np.max(np.abs(conv_result))

# Setup figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])  # Correlation animation top
ax2 = fig.add_subplot(gs[1, 0])  # Correlation product
ax3 = fig.add_subplot(gs[2, 0])  # Correlation result
ax4 = fig.add_subplot(gs[0, 1])  # Convolution animation top
ax5 = fig.add_subplot(gs[1, 1])  # Convolution product
ax6 = fig.add_subplot(gs[2, 1])  # Convolution result

# Initialize plots
line_sq_corr, = ax1.plot(t, square_pulse, 'b-', linewidth=2, label='Square')
line_saw_corr, = ax1.plot(t, sawtooth_pulse, 'r-', linewidth=2, label='Sawtooth (sliding)')
ax1.set_title('Correlation: Sliding Sawtooth')
ax1.set_ylabel('Amplitude')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-0.2, 1.3])

line_product_corr, = ax2.plot(t, np.zeros_like(t), 'g-', linewidth=2)
ax2.set_title('Product (overlap)')
ax2.set_ylabel('Product')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
fill_corr = ax2.fill_between(t, 0, np.zeros_like(t), alpha=0.3, color='g')

lags = signal.correlation_lags(len(square_pulse), len(sawtooth_pulse), mode='same')
lag_time = lags / sample_rate
line_result_corr, = ax3.plot([], [], 'g-', linewidth=2)
point_corr, = ax3.plot([], [], 'ro', markersize=8)
ax3.plot(lag_time, cross_corr_norm, 'g--', linewidth=1, alpha=0.3, label='Full result')
ax3.set_title('Correlation Result (building)')
ax3.set_xlabel('Lag Time (seconds)')
ax3.set_ylabel('Correlation')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim([lag_time[0], lag_time[-1]])
ax3.set_ylim([np.min(cross_corr_norm) - 0.1, np.max(cross_corr_norm) + 0.1])

# Convolution plots (sawtooth is flipped)
sawtooth_flipped = np.flip(sawtooth_pulse)
line_sq_conv, = ax4.plot(t, square_pulse, 'b-', linewidth=2, label='Square')
line_saw_conv, = ax4.plot(t, sawtooth_flipped, 'r-', linewidth=2, label='Sawtooth (flipped & sliding)')
ax4.set_title('Convolution: Flipped & Sliding Sawtooth')
ax4.set_ylabel('Amplitude')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([-0.2, 1.3])

line_product_conv, = ax5.plot(t, np.zeros_like(t), 'm-', linewidth=2)
ax5.set_title('Product (overlap)')
ax5.set_ylabel('Product')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='k', linewidth=0.5)
fill_conv = ax5.fill_between(t, 0, np.zeros_like(t), alpha=0.3, color='m')

line_result_conv, = ax6.plot([], [], 'm-', linewidth=2)
point_conv, = ax6.plot([], [], 'ro', markersize=8)
ax6.plot(t, conv_result_norm, 'm--', linewidth=1, alpha=0.3, label='Full result')
ax6.set_title('Convolution Result (building)')
ax6.set_xlabel('Time (seconds)')
ax6.set_ylabel('Convolution')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xlim([t[0], t[-1]])
ax6.set_ylim([np.min(conv_result_norm) - 0.1, np.max(conv_result_norm) + 0.1])

# Animation data
num_frames = 150
shift_range = np.linspace(-len(t)//2, len(t)//2, num_frames).astype(int)

# Store results for animation
corr_results = []
conv_results = []
corr_lags = []
conv_times = []

def init():
    return (line_saw_corr, line_product_corr, line_result_corr, point_corr,
            line_saw_conv, line_product_conv, line_result_conv, point_conv)

def animate(frame):
    shift = shift_range[frame]
    
    # Correlation: slide sawtooth
    saw_shifted = np.roll(sawtooth_pulse, shift)
    if shift > 0:
        saw_shifted[:shift] = 0
    elif shift < 0:
        saw_shifted[shift:] = 0
    
    line_saw_corr.set_ydata(saw_shifted)
    
    # Product for correlation
    product_corr = square_pulse * saw_shifted
    line_product_corr.set_ydata(product_corr)
    
    # Update fill
    global fill_corr
    fill_corr.remove()
    fill_corr = ax2.fill_between(t, 0, product_corr, alpha=0.3, color='g')
    
    # Correlation value (sum of product)
    corr_value = np.sum(product_corr)
    corr_value_norm = corr_value / np.max(np.abs(cross_corr))
    corr_results.append(corr_value_norm)
    current_lag = shift / sample_rate
    corr_lags.append(current_lag)
    
    line_result_corr.set_data(corr_lags, corr_results)
    point_corr.set_data([current_lag], [corr_value_norm])
    
    # Convolution: slide flipped sawtooth
    saw_flipped_shifted = np.roll(sawtooth_flipped, shift)
    if shift > 0:
        saw_flipped_shifted[:shift] = 0
    elif shift < 0:
        saw_flipped_shifted[shift:] = 0
    
    line_saw_conv.set_ydata(saw_flipped_shifted)
    
    # Product for convolution
    product_conv = square_pulse * saw_flipped_shifted
    line_product_conv.set_ydata(product_conv)
    
    # Update fill
    global fill_conv
    fill_conv.remove()
    fill_conv = ax5.fill_between(t, 0, product_conv, alpha=0.3, color='m')
    
    # Convolution value (sum of product)
    conv_value = np.sum(product_conv)
    conv_value_norm = conv_value / np.max(np.abs(conv_result))
    conv_results.append(conv_value_norm)
    
    # Calculate proper time position for convolution
    center_idx = len(t) // 2
    current_idx = center_idx + shift
    if 0 <= current_idx < len(t):
        current_time = t[current_idx]
    else:
        current_time = t[min(max(current_idx, 0), len(t)-1)]
    conv_times.append(current_time)
    
    line_result_conv.set_data(conv_times, conv_results)
    point_conv.set_data([current_time], [conv_value_norm])
    
    return (line_saw_corr, line_product_corr, line_result_corr, point_corr,
            line_saw_conv, line_product_conv, line_result_conv, point_conv, fill_corr, fill_conv)

anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, 
                    interval=50, blit=False, repeat=True)

plt.show()

print("Animation Guide:")
print("\nLEFT COLUMN (Correlation):")
print("- Top: Sawtooth slides across square wave")
print("- Middle: Shows the product (overlap) at each position")
print("- Bottom: Correlation result builds up as sawtooth slides")
print("\nRIGHT COLUMN (Convolution):")
print("- Top: Sawtooth is FLIPPED then slides across square wave")
print("- Middle: Shows the product (overlap) at each position")
print("- Bottom: Convolution result builds up")
print("\nKey difference: Convolution flips one signal before sliding!")
