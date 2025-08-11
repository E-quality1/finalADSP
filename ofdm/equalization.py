"""
Channel Estimation and Equalization
"""
import numpy as np
import logging
from ofdm.system import params

def ls_channel_estimation(rx_pilots, tx_pilots):
    """Least Squares channel estimation on pilot subcarriers"""
    return rx_pilots / tx_pilots

def mmse_channel_estimation(ls_est, noise_var, channel_var=1.0):
    """MMSE channel estimation from LS estimate"""
    return (channel_var * ls_est) / (channel_var + noise_var)

def zf_equalizer(rx_symbol, h_est):
    """Zero Forcing per-tone equalization"""
    return rx_symbol / h_est

def mmse_equalizer(rx_symbol, h_est, noise_var, symbol_var=1.0):
    """MMSE per-tone equalization"""
    return (np.conj(h_est) / (np.abs(h_est)**2 + noise_var/symbol_var)) * rx_symbol

def remove_cp_and_fft(rx_signal, timing_offset):
    """Isolates the first data symbol, removes CP, and performs FFT."""
    # The Schmidl & Cox peak occurs at the end of the first half of the preamble.
    # The full preamble has length N. The data symbol starts right after.
    # Therefore, the start of the data symbol's CP is at the timing_offset.
    data_start = timing_offset
    symbol_with_cp = rx_signal[data_start : data_start + params.N + params.CP]
    symbol_without_cp = symbol_with_cp[params.CP:]
    rx_symbol_freq_natural = np.fft.fft(symbol_without_cp, params.N)
    # Shift to center the DC frequency for easier processing
    rx_symbol_freq_centered = np.fft.fftshift(rx_symbol_freq_natural)
    return rx_symbol_freq_centered

def estimate_integer_cfo_pilots(rx_symbol_freq):
    """Estimates integer CFO by correlating the ideal pilot template against shifted versions of the received spectrum."""
    max_cfo = 8  # Search range for integer CFO

    # Create ideal pilot template for centered spectrum
    # After fftshift, pilots move to new positions
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    # Convert to positions after fftshift: index k -> (k + N/2) % N
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    
    ideal_pilot_template = np.zeros(params.N, dtype=complex)
    ideal_pilot_template[pilot_indices_after_fftshift] = params.pilot_value

    correlations = []
    for shift in range(-max_cfo, max_cfo + 1):
        # Correlate a shifted version of the received signal with the ideal pilot template
        shifted_rx_freq = np.roll(rx_symbol_freq, shift)
        # Correlate by taking the dot product of the received signal with the *conjugated* pilot template
        corr = np.abs(np.vdot(ideal_pilot_template, shifted_rx_freq))
        correlations.append(corr)
    
    # The best integer CFO estimate is the negative of the shift that maximizes correlation.
    best_shift = np.argmax(correlations) - max_cfo
    est_int_cfo = -best_shift
    return est_int_cfo

def estimate_and_correct_cfo(rx_signal, timing_offset):
    """Estimates and corrects CFO using a robust two-stage frequency-domain method."""
    # 1. Remove CP and perform FFT to get the centered frequency spectrum
    rx_symbol_freq_centered = remove_cp_and_fft(rx_signal, timing_offset)

    # 2. Estimate and Correct Integer CFO on the centered spectrum
    est_int_cfo = estimate_integer_cfo_pilots(rx_symbol_freq_centered)
    # Correct the integer CFO by shifting in the opposite direction of the offset
    rx_symbol_freq_centered_corrected = np.roll(rx_symbol_freq_centered, -est_int_cfo)

    # 3. Estimate Fractional CFO from the centered, integer-corrected symbol
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    # Convert to positions after fftshift: index k -> (k + N/2) % N
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    rx_pilots = rx_symbol_freq_centered_corrected[pilot_indices_after_fftshift]
    
    # Get pilot values and their indices *in the centered spectrum*
    first_pilot_val, last_pilot_val = rx_pilots[0], rx_pilots[-1]
    # Use the original pilot carriers for span calculation
    pilot_carriers = np.array([-21, -7, 7, 21])
    first_pilot_idx_centered, last_pilot_idx_centered = pilot_carriers[0], pilot_carriers[-1]
    pilot_span = last_pilot_idx_centered - first_pilot_idx_centered
    
    # Robust phase difference calculation with unwrapping
    angle_first = np.angle(first_pilot_val)
    angle_last = np.angle(last_pilot_val)
    phase_diff = angle_last - angle_first
    # Unwrap the phase to handle cases where it exceeds +/- pi
    if phase_diff > np.pi:
        phase_diff -= 2 * np.pi
    elif phase_diff < -np.pi:
        phase_diff += 2 * np.pi
    est_frac_cfo = -phase_diff * params.N / (2 * np.pi * pilot_span)

    # 4. Apply fractional correction as a phase ramp across the centered, integer-corrected symbol
    subcarrier_indices = np.arange(params.N)
    # The subcarrier indices for the centered spectrum range from -N/2 to N/2-1
    centered_indices = np.arange(-params.N//2, params.N//2)
    correction_frac = np.exp(-1j * 2 * np.pi * est_frac_cfo * centered_indices / params.N)
    corrected_symbol_full = rx_symbol_freq_centered_corrected * correction_frac

    # 5. Log results
    cfo_est_total = est_int_cfo + est_frac_cfo
    logging.info(f"Total CFO Estimate: {cfo_est_total:.4f} (Int: {est_int_cfo}, Frac: {est_frac_cfo:.4f})")

    return corrected_symbol_full, cfo_est_total


# Test block for visualization
if __name__ == "__main__":
    import logging
    import matplotlib.pyplot as plt
    # Use relative imports for script execution within a package
    from ofdm.transmitter import generate_ofdm_frame
    from ofdm.channel import apply_channel_effects
    from ofdm.synchronization import schmidl_cox_metric

    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # 1. Generate a transmitted signal (preamble + 1 data symbol)
    tx_frame = generate_ofdm_frame()

    # 2. Simulate channel effects
    delay_s = 2.15e-6  # 2.15 us, corresponds to 43 samples at 20MHz
    snr_db = 15
    cfo = 1.1 * params.subcarrier_spacing
    rx_signal, actual_offset = apply_channel_effects(
        tx_frame, cfo=cfo, snr_db=snr_db, delay_s=delay_s, apply_fading=True
    )
    logging.info(f"Signal generated with offset {actual_offset}, SNR={snr_db}dB, CFO={cfo / 1e3:.2f}kHz")

    # 3. Perform synchronization (for comparison, but we use perfect offset for test)
    metric, P, R = schmidl_cox_metric(rx_signal, params.N // 2)
    est_timing_offset = np.argmax(metric)
    logging.info(f"Estimated timing offset: {est_timing_offset} (vs. actual: {actual_offset})")

    # 4. Process the first data symbol using PERFECT timing
    logging.info(f"--- Using PERFECT timing offset ({actual_offset}) for diagnostics ---")
    
    # 5. Full CFO Estimation and Correction (Two-Stage)
    corrected_symbol_full, total_cfo_est = estimate_and_correct_cfo(rx_signal, actual_offset)
    logging.info(f"Actual Total CFO: {cfo / params.subcarrier_spacing:.4f}, Estimated Total CFO: {total_cfo_est:.4f}")

    # --- Plots ---
    data_indices = np.where(~params.pilot_pattern)[0]
    rx_data_symbols = corrected_symbol_full[data_indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(rx_data_symbols), np.imag(rx_data_symbols), alpha=0.7)
    plt.title('Received Constellation after Full CFO Correction (with Perfect Timing)')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('rx_constellation_after_full_cfo.png')
    plt.close()
    logging.info("Plot saved to rx_constellation_after_full_cfo.png")
