"""
Channel Models: Rayleigh Fading, AWGN, CFO
"""
import numpy as np
from ofdm.system import params

def rayleigh_channel(num_taps=None, rms_delay=None):
    if num_taps is None:
        num_taps = params.num_taps
    if rms_delay is None:
        rms_delay = params.delay_spread
    delays = np.linspace(0, rms_delay, num_taps)
    power_profile = np.exp(-delays / rms_delay)
    power_profile /= np.sum(power_profile)
    taps = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
    taps *= np.sqrt(power_profile)
    return taps

def apply_channel(signal, taps):
    return np.convolve(signal, taps, mode='full')[:len(signal)]

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

def add_cfo(signal, cfo, fs):
    n = np.arange(len(signal))
    return signal * np.exp(1j*2*np.pi*cfo*n/fs)

def apply_channel_effects(tx_signal, cfo, snr_db, delay_s, apply_fading=True):
    """Applies timing delay, channel, CFO, and noise."""
    # 1. Add timing delay
    delay_samples = int(delay_s * params.sampling_rate)
    delayed_signal = np.concatenate([np.zeros(delay_samples), tx_signal])

    # 2. Apply Rayleigh Fading Channel (if enabled)
    if apply_fading:
        channel_taps = rayleigh_channel()
        faded_signal = apply_channel(delayed_signal, channel_taps)
    else:
        faded_signal = delayed_signal

    # 3. Add CFO
    cfo_signal = add_cfo(faded_signal, cfo, params.sampling_rate)

    # 4. Add AWGN
    rx_signal = add_awgn(cfo_signal, snr_db)

    return rx_signal, delay_samples


# Test block for visualization
if __name__ == "__main__":
    import logging
    import matplotlib.pyplot as plt
    from transmitter import generate_preamble, ofdm_symbol_mapper, insert_pilots, ifft_with_cp

    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # 1. Generate a test OFDM frame (preamble + 1 data symbol)
    preamble = generate_preamble()
    num_data = np.sum(~params.pilot_pattern)
    bits = np.random.randint(0, 2, params.bits_per_symbol * num_data)
    symbols = ofdm_symbol_mapper(bits)
    ofdm_symbol = insert_pilots(symbols)
    data_signal = ifft_with_cp(ofdm_symbol)
    tx_signal = np.concatenate([preamble, data_signal])
    logging.info(f"Generated transmit signal of length {len(tx_signal)}.")

    # 2. Apply Rayleigh Fading Channel
    channel_taps = rayleigh_channel()
    faded_signal = apply_channel(tx_signal, channel_taps)
    logging.info(f"Applied {len(channel_taps)}-tap Rayleigh channel.")

    # 3. Add AWGN
    snr_db = 15  # High SNR for clear visualization
    noisy_signal = add_awgn(faded_signal, snr_db)
    logging.info(f"Added AWGN for SNR = {snr_db} dB.")

    # 4. Add CFO
    cfo = 0.1 * params.subcarrier_spacing # 10% of subcarrier spacing
    rx_signal = add_cfo(noisy_signal, cfo, params.sampling_rate)
    logging.info(f"Added CFO of {cfo/1e3:.2f} kHz.")

    # --- Plots ---
    # Plot Channel Impulse Response
    plt.figure()
    plt.stem(np.abs(channel_taps))
    plt.title('Rayleigh Channel Impulse Response')
    plt.xlabel('Tap Index')
    plt.ylabel('Magnitude')
    plt.savefig('channel_impulse_response.png')
    plt.close()

    # Plot signal at each stage
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(np.real(tx_signal), label='Original')
    axs[0].set_title('Original Transmitted Signal')
    axs[0].legend()

    axs[1].plot(np.real(faded_signal), label='Faded')
    axs[1].set_title('Signal after Rayleigh Fading')
    axs[1].legend()

    axs[2].plot(np.real(rx_signal), label='Faded + Noise + CFO')
    axs[2].set_title(f'Final Received Signal (SNR={snr_db}dB)')
    axs[2].legend()
    plt.xlabel('Sample Index')
    plt.tight_layout()
    plt.savefig('channel_effects_on_signal.png')
    plt.close()
