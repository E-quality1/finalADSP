"""
Timing and Frequency Synchronization (Schmidl & Cox)
"""
import numpy as np
from ofdm.system import params

def schmidl_cox_metric(rx_signal, window):
    """Compute Schmidl & Cox timing metric over rx_signal"""
    N = window
    M = len(rx_signal) - 2*N
    P = np.zeros(M, dtype=complex)
    R = np.zeros(M)
    metric = np.zeros(M)
    for d in range(M):
        P[d] = np.sum(rx_signal[d:d+N] * np.conj(rx_signal[d+N:d+2*N]))
        R[d] = np.sum(np.abs(rx_signal[d+N:d+2*N])**2)
        if R[d] > 0:
            metric[d] = np.abs(P[d])**2 / (R[d]**2)
        else:
            metric[d] = 0
    return metric, P, R

def estimate_fractional_cfo(P, d_est):
    """Estimate fractional CFO from Schmidl & Cox correlation output"""
    # Use the phase of the correlation term at the estimated timing instant
    return np.angle(P[d_est]) / (np.pi) # Note: Schmidl & Cox formula has 2*pi*L/N, L=N/2 -> pi

# Test block for visualization
if __name__ == "__main__":
    import logging
    import matplotlib.pyplot as plt
    from transmitter import generate_preamble, ofdm_symbol_mapper, insert_pilots, ifft_with_cp
    from channel import apply_channel, add_awgn, add_cfo, rayleigh_channel

    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # 1. Generate a test OFDM frame
    preamble = generate_preamble()
    num_data = np.sum(~params.pilot_pattern)
    bits = np.random.randint(0, 2, params.bits_per_symbol * num_data)
    symbols = ofdm_symbol_mapper(bits)
    ofdm_symbol = insert_pilots(symbols)
    data_signal = ifft_with_cp(ofdm_symbol)
    tx_signal = np.concatenate([preamble, data_signal])

    # 2. Add a random timing offset (delay)
    timing_offset = np.random.randint(20, 50)
    delayed_signal = np.concatenate([np.zeros(timing_offset), tx_signal])
    logging.info(f"Introduced a timing offset of {timing_offset} samples.")

    # 3. Pass through channel
    channel_taps = rayleigh_channel()
    faded_signal = apply_channel(delayed_signal, channel_taps)
    snr_db = 15
    noisy_signal = add_awgn(faded_signal, snr_db)
    cfo = 0.1 * params.subcarrier_spacing
    rx_signal = add_cfo(noisy_signal, cfo, params.sampling_rate)
    logging.info(f"Signal passed through channel (SNR={snr_db}dB, CFO={cfo/1e3:.2f}kHz)")

    # 4. Perform timing synchronization
    metric, P, R = schmidl_cox_metric(rx_signal, params.N // 2)
    est_timing_offset = np.argmax(metric)
    logging.info(f"Estimated timing offset: {est_timing_offset} samples.")

    # 5. Estimate fractional CFO
    est_frac_cfo = estimate_fractional_cfo(P, est_timing_offset)
    logging.info(f"Actual fractional CFO: {cfo / params.subcarrier_spacing:.4f}")
    logging.info(f"Estimated fractional CFO: {est_frac_cfo:.4f}")

    # --- Plots ---
    plt.figure(figsize=(12, 6))
    plt.plot(metric)
    plt.axvline(x=est_timing_offset, color='r', linestyle='--', label=f'Estimated Peak ({est_timing_offset})')
    plt.axvline(x=timing_offset, color='g', linestyle=':', label=f'Actual Start ({timing_offset})')
    plt.title('Schmidl & Cox Timing Metric')
    plt.xlabel('Sample Index (d)')
    plt.ylabel('Metric M(d)')
    plt.legend()
    plt.grid(True)
    plt.savefig('timing_sync_metric.png')
    plt.close()
