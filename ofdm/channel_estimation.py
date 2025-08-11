"""
OFDM Channel Estimation Module

This module implements channel estimation techniques for OFDM systems:
1. Least Squares (LS) channel estimation using pilot subcarriers
2. Minimum Mean Square Error (MMSE) channel estimation
3. Channel interpolation for data subcarriers
4. Channel quality metrics and analysis

References:
- Coleri et al. (2002): "Channel estimation techniques based on pilot arrangement in OFDM systems"
- van de Beek et al. (1995): "On channel estimation in OFDM systems"
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy import interpolate
from ofdm.system import params

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def ls_channel_estimation(rx_pilots, tx_pilots):
    """
    Least Squares (LS) channel estimation using pilot subcarriers.
    
    The LS estimate is: H_LS = Y_p / X_p
    where Y_p are received pilots and X_p are transmitted pilots.
    
    Args:
        rx_pilots: Received pilot symbols (complex array)
        tx_pilots: Transmitted pilot symbols (complex array)
    
    Returns:
        channel_est_pilots: Channel estimates at pilot locations (complex array)
    """
    # LS estimation: H = Y / X
    channel_est_pilots = rx_pilots / tx_pilots
    
    logging.info(f"LS channel estimation: {len(channel_est_pilots)} pilot estimates computed")
    return channel_est_pilots

def mmse_channel_estimation(rx_pilots, tx_pilots, snr_db=15.0, channel_length=5):
    """
    Minimum Mean Square Error (MMSE) channel estimation using pilot subcarriers.
    
    The MMSE estimate incorporates noise statistics and channel correlation:
    H_MMSE = R_hh * (R_hh + (sigma_n^2 / sigma_s^2) * I)^(-1) * H_LS
    
    Args:
        rx_pilots: Received pilot symbols (complex array)
        tx_pilots: Transmitted pilot symbols (complex array)
        snr_db: Signal-to-noise ratio in dB
        channel_length: Expected channel length in samples
    
    Returns:
        channel_est_pilots: MMSE channel estimates at pilot locations (complex array)
    """
    # First compute LS estimate
    h_ls = ls_channel_estimation(rx_pilots, tx_pilots)
    
    # Convert SNR to linear scale
    snr_linear = 10**(snr_db / 10.0)
    noise_var = 1.0 / snr_linear
    
    # Create channel correlation matrix (simplified exponential model)
    num_pilots = len(rx_pilots)
    pilot_indices = np.where(params.pilot_pattern)[0]
    
    # After fftshift, pilots are at different positions
    pilot_indices_after_fftshift = (pilot_indices + params.N//2) % params.N
    
    # Compute correlation matrix based on pilot separation
    R_hh = np.zeros((num_pilots, num_pilots), dtype=complex)
    for i in range(num_pilots):
        for j in range(num_pilots):
            # Distance between pilots in frequency domain
            freq_dist = abs(pilot_indices_after_fftshift[i] - pilot_indices_after_fftshift[j])
            # Exponential correlation model
            R_hh[i, j] = np.exp(-freq_dist / channel_length)
    
    # MMSE estimation
    R_nn = noise_var * np.eye(num_pilots)  # Noise correlation matrix
    inv_matrix = np.linalg.inv(R_hh + R_nn)
    h_mmse = R_hh @ inv_matrix @ h_ls
    
    logging.info(f"MMSE channel estimation: {len(h_mmse)} pilot estimates computed (SNR={snr_db}dB)")
    return h_mmse

def interpolate_channel(channel_est_pilots, method='linear'):
    """
    Interpolate channel estimates from pilot subcarriers to all subcarriers.
    
    Args:
        channel_est_pilots: Channel estimates at pilot locations (complex array)
        method: Interpolation method ('linear', 'cubic', 'spline')
    
    Returns:
        channel_est_full: Channel estimates for all subcarriers (complex array)
    """
    # Get pilot indices after fftshift
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    
    # Create full subcarrier index array (centered)
    all_indices = np.arange(params.N)
    
    # Interpolate real and imaginary parts separately
    if method == 'linear':
        # Linear interpolation
        real_interp = np.interp(all_indices, pilot_indices_after_fftshift, 
                               np.real(channel_est_pilots))
        imag_interp = np.interp(all_indices, pilot_indices_after_fftshift, 
                               np.imag(channel_est_pilots))
    
    elif method == 'cubic':
        # Cubic spline interpolation
        if len(channel_est_pilots) >= 4:  # Need at least 4 points for cubic
            real_spline = interpolate.CubicSpline(pilot_indices_after_fftshift, 
                                                 np.real(channel_est_pilots), 
                                                 bc_type='natural')
            imag_spline = interpolate.CubicSpline(pilot_indices_after_fftshift, 
                                                 np.imag(channel_est_pilots), 
                                                 bc_type='natural')
            real_interp = real_spline(all_indices)
            imag_interp = imag_spline(all_indices)
        else:
            # Fall back to linear if not enough points
            real_interp = np.interp(all_indices, pilot_indices_after_fftshift, 
                                   np.real(channel_est_pilots))
            imag_interp = np.interp(all_indices, pilot_indices_after_fftshift, 
                                   np.imag(channel_est_pilots))
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Combine real and imaginary parts
    channel_est_full = real_interp + 1j * imag_interp
    
    logging.info(f"Channel interpolation ({method}): {len(channel_est_full)} subcarrier estimates")
    return channel_est_full

def estimate_channel_from_ofdm_symbol(rx_symbol_freq, method='ls', snr_db=15.0):
    """
    Complete channel estimation from a received OFDM symbol.
    
    Args:
        rx_symbol_freq: Received OFDM symbol in frequency domain (after CFO correction)
        method: Channel estimation method ('ls' or 'mmse')
        snr_db: Signal-to-noise ratio in dB (for MMSE)
    
    Returns:
        channel_est_full: Channel estimates for all subcarriers (complex array)
        channel_est_pilots: Channel estimates at pilot locations (complex array)
    """
    # Extract pilot subcarriers from received symbol
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    rx_pilots = rx_symbol_freq[pilot_indices_after_fftshift]
    
    # Known transmitted pilot values
    tx_pilots = np.full(len(rx_pilots), params.pilot_value, dtype=complex)
    
    # Perform channel estimation at pilot locations
    if method.lower() == 'ls':
        channel_est_pilots = ls_channel_estimation(rx_pilots, tx_pilots)
    elif method.lower() == 'mmse':
        channel_est_pilots = mmse_channel_estimation(rx_pilots, tx_pilots, snr_db)
    else:
        raise ValueError(f"Unknown channel estimation method: {method}")
    
    # Interpolate to all subcarriers
    channel_est_full = interpolate_channel(channel_est_pilots, method='linear')
    
    return channel_est_full, channel_est_pilots

def analyze_channel_quality(channel_est_full):
    """
    Analyze channel quality metrics.
    
    Args:
        channel_est_full: Channel estimates for all subcarriers
    
    Returns:
        metrics: Dictionary containing channel quality metrics
    """
    metrics = {}
    
    # Channel magnitude statistics
    channel_mag = np.abs(channel_est_full)
    metrics['mean_magnitude'] = np.mean(channel_mag)
    metrics['std_magnitude'] = np.std(channel_mag)
    metrics['min_magnitude'] = np.min(channel_mag)
    metrics['max_magnitude'] = np.max(channel_mag)
    
    # Channel phase statistics
    channel_phase = np.angle(channel_est_full)
    metrics['mean_phase'] = np.mean(channel_phase)
    metrics['std_phase'] = np.std(channel_phase)
    
    # Frequency selectivity (variation across subcarriers)
    metrics['frequency_selectivity'] = np.std(channel_mag) / np.mean(channel_mag)
    
    # Condition number (for equalization difficulty)
    metrics['condition_number'] = np.max(channel_mag) / np.max([np.min(channel_mag), 1e-10])
    
    logging.info(f"Channel analysis: mean_mag={metrics['mean_magnitude']:.3f}, "
                f"freq_selectivity={metrics['frequency_selectivity']:.3f}")
    
    return metrics

def plot_channel_estimate(channel_est_full, title="Channel Estimate", save_path=None):
    """
    Plot channel estimate magnitude and phase.
    
    Args:
        channel_est_full: Channel estimates for all subcarriers
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    subcarriers = np.arange(-params.N//2, params.N//2)
    
    # Magnitude plot
    ax1.plot(subcarriers, np.abs(channel_est_full), 'b-', linewidth=2)
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'{title} - Magnitude Response')
    ax1.grid(True, alpha=0.3)
    
    # Mark pilot locations
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_carriers = pilot_indices_natural - params.N//2  # Convert to centered
    pilot_mags = np.abs(channel_est_full)[pilot_indices_natural]
    ax1.scatter(pilot_carriers, pilot_mags, color='red', s=50, zorder=5, label='Pilots')
    ax1.legend()
    
    # Phase plot
    ax2.plot(subcarriers, np.angle(channel_est_full), 'r-', linewidth=2)
    ax2.set_xlabel('Subcarrier Index')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title(f'{title} - Phase Response')
    ax2.grid(True, alpha=0.3)
    
    # Mark pilot locations
    pilot_phases = np.angle(channel_est_full)[pilot_indices_natural]
    ax2.scatter(pilot_carriers, pilot_phases, color='red', s=50, zorder=5, label='Pilots')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Channel estimate plot saved to {save_path}")
    
    return fig

# Test block for channel estimation validation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Import required modules for testing
    from ofdm.transmitter import qam_mod, insert_pilots, ifft_with_cp
    from ofdm.channel import apply_channel_effects
    from ofdm.equalization import estimate_and_correct_cfo, remove_cp_and_fft
    
    print("=" * 60)
    print("CHANNEL ESTIMATION MODULE TEST")
    print("=" * 60)
    
    # Generate test OFDM signal
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    data_bits = np.random.randint(0, 2, num_data_bits)
    data_symbols = qam_mod(data_bits)
    
    # Create OFDM symbol
    ofdm_symbol_with_pilots = insert_pilots(data_symbols)
    time_domain_signal = ifft_with_cp(ofdm_symbol_with_pilots)
    
    # Apply channel effects
    rx_signal, delay_samples = apply_channel_effects(time_domain_signal, 
                                                    cfo=0.0, snr_db=15, 
                                                    delay_s=0.0, apply_fading=True)
    
    # Receiver processing
    rx_symbol_freq_corrected, _ = estimate_and_correct_cfo(rx_signal, 0)
    
    # Test both LS and MMSE channel estimation
    for method in ['ls', 'mmse']:
        print(f"\nTesting {method.upper()} channel estimation:")
        
        channel_est_full, channel_est_pilots = estimate_channel_from_ofdm_symbol(
            rx_symbol_freq_corrected, method=method, snr_db=15)
        
        # Analyze channel quality
        metrics = analyze_channel_quality(channel_est_full)
        print(f"  Mean magnitude: {metrics['mean_magnitude']:.3f}")
        print(f"  Frequency selectivity: {metrics['frequency_selectivity']:.3f}")
        print(f"  Condition number: {metrics['condition_number']:.1f}")
        
        # Plot channel estimate
        plot_channel_estimate(channel_est_full, 
                            title=f"{method.upper()} Channel Estimate",
                            save_path=f"channel_estimate_{method}.png")
    
    print("\n" + "=" * 60)
    print("CHANNEL ESTIMATION TEST COMPLETE")
    print("=" * 60)
