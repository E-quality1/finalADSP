"""
Main OFDM Receiver Simulation Script

This script integrates all implemented modules to simulate a complete
64-QAM OFDM receiver chain following IEEE 802.11-class specifications.

Components tested:
1. OFDM Transmitter (QAM modulation, pilot insertion, IFFT+CP)
2. Channel Model (Rayleigh fading, AWGN, CFO, timing offset)
3. Timing Synchronization (Schmidl & Cox)
4. CFO Estimation and Correction (two-stage: integer + fractional)
5. Channel Estimation (LS and MMSE)
6. Equalization (ZF and MMSE)
7. Performance Analysis (BER, constellation plots)
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Import OFDM modules
from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp, generate_preamble, generate_ofdm_frame
from ofdm.channel import apply_channel_effects
from ofdm.synchronization import schmidl_cox_metric
from ofdm.equalization import estimate_and_correct_cfo, remove_cp_and_fft
from ofdm.channel_estimation import estimate_channel_from_ofdm_symbol, plot_channel_estimate
from ofdm.equalization import ls_channel_estimation, zf_equalizer, mmse_equalizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def generate_test_frame():
    """Generate a complete OFDM test frame with preamble and data."""
    # Generate random data bits
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    data_bits = np.random.randint(0, 2, num_data_bits)
    
    # Generate complete OFDM frame
    tx_frame = generate_ofdm_frame()
    
    logging.info(f"Generated OFDM frame: {len(tx_frame)} samples, {num_data_bits} data bits")
    return tx_frame, data_bits

def timing_synchronization(rx_signal):
    """Perform timing synchronization using Schmidl & Cox method."""
    # Use half the OFDM symbol length as the correlation window
    window = params.N // 2
    
    # Compute Schmidl & Cox metric
    metric, P, R = schmidl_cox_metric(rx_signal, window)
    
    # Find the peak of the timing metric
    estimated_timing_offset = np.argmax(metric)
    
    logging.info(f"Timing sync: peak at sample {estimated_timing_offset}, metric={metric[estimated_timing_offset]:.4f}")
    
    return estimated_timing_offset

def simulate_channel_impairments(tx_signal, snr_db=15, cfo_hz=343.75e3, timing_offset_samples=43):
    """Apply realistic channel impairments to the transmitted signal."""
    # Convert timing offset to delay in seconds
    delay_s = timing_offset_samples / params.sampling_rate
    
    # Apply channel effects
    rx_signal, actual_delay_samples = apply_channel_effects(
        tx_signal, cfo=cfo_hz, snr_db=snr_db, delay_s=delay_s, apply_fading=True)
    
    logging.info(f"Applied channel: SNR={snr_db}dB, CFO={cfo_hz/1e3:.1f}kHz, "
                f"timing_offset={actual_delay_samples} samples")
    
    return rx_signal, actual_delay_samples

def full_receiver_chain(rx_signal, actual_timing_offset, snr_db=15):
    """Process received signal through complete receiver chain."""
    results = {}
    
    # 1. Timing Synchronization
    logging.info("Step 1: Timing Synchronization")
    estimated_timing_offset = timing_synchronization(rx_signal)
    timing_error = estimated_timing_offset - actual_timing_offset
    results['timing_offset_est'] = estimated_timing_offset
    results['timing_offset_actual'] = actual_timing_offset
    results['timing_error'] = timing_error
    
    logging.info(f"Timing sync: estimated={estimated_timing_offset}, "
                f"actual={actual_timing_offset}, error={timing_error}")
    
    # Use perfect timing for now to isolate other issues
    timing_offset_to_use = actual_timing_offset
    logging.info(f"Using perfect timing offset ({timing_offset_to_use}) for downstream processing")
    
    # 2. CFO Estimation and Correction
    logging.info("Step 2: CFO Estimation and Correction")
    rx_symbol_freq_corrected, cfo_estimates = estimate_and_correct_cfo(rx_signal, timing_offset_to_use)
    results['cfo_estimates'] = cfo_estimates
    
    # 3. Channel Estimation
    logging.info("Step 3: Channel Estimation")
    channel_est_full_ls, channel_est_pilots_ls = estimate_channel_from_ofdm_symbol(
        rx_symbol_freq_corrected, method='ls', snr_db=snr_db)
    
    channel_est_full_mmse, channel_est_pilots_mmse = estimate_channel_from_ofdm_symbol(
        rx_symbol_freq_corrected, method='mmse', snr_db=snr_db)
    
    results['channel_est_ls'] = channel_est_full_ls
    results['channel_est_mmse'] = channel_est_full_mmse
    
    # 4. Equalization
    logging.info("Step 4: Equalization")
    
    # ZF Equalization with LS channel estimate
    equalized_zf_ls = zf_equalizer(rx_symbol_freq_corrected, channel_est_full_ls)
    
    # MMSE Equalization with MMSE channel estimate
    noise_var = 10**(-snr_db/10)  # Convert SNR to noise variance
    equalized_mmse_mmse = mmse_equalizer(rx_symbol_freq_corrected, channel_est_full_mmse, noise_var)
    
    results['equalized_zf_ls'] = equalized_zf_ls
    results['equalized_mmse_mmse'] = equalized_mmse_mmse
    
    return results

def analyze_performance(results, original_data_bits):
    """Analyze receiver performance metrics."""
    performance = {}
    
    # Extract data subcarriers (non-pilot positions after fftshift)
    # Use correct pilot indices for IEEE 802.11 positions [-21, -7, 7, 21]
    pilot_indices_natural = np.where(params.pilot_pattern)[0]  # [11, 25, 39, 53]
    # After fftshift, these natural indices correspond to the correct centered positions
    pilot_indices_centered = pilot_indices_natural  # No shift needed!
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_centered] = False
    
    # Test different equalization methods
    equalization_methods = {
        'ZF+LS': results['equalized_zf_ls'],
        'MMSE+MMSE': results['equalized_mmse_mmse']
    }
    
    for method_name, equalized_symbols in equalization_methods.items():
        # Extract data symbols
        data_symbols_eq = equalized_symbols[data_mask]
        
        # Demodulate to bits
        try:
            demod_bits = qam_demod(data_symbols_eq)
            
            # Calculate BER (if lengths match)
            if len(demod_bits) == len(original_data_bits):
                bit_errors = np.sum(demod_bits != original_data_bits)
                ber = bit_errors / len(original_data_bits)
                performance[f'BER_{method_name}'] = ber
                logging.info(f"{method_name} BER: {ber:.6f} ({bit_errors}/{len(original_data_bits)} errors)")
            else:
                logging.warning(f"{method_name}: Length mismatch - demod:{len(demod_bits)}, orig:{len(original_data_bits)}")
                performance[f'BER_{method_name}'] = float('inf')
        
        except Exception as e:
            logging.error(f"Error in {method_name} demodulation: {e}")
            performance[f'BER_{method_name}'] = float('inf')
    
    return performance

def plot_results(results, save_plots=True):
    """Generate comprehensive result plots."""
    
    # 1. Channel Estimates Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    subcarriers = np.arange(-params.N//2, params.N//2)
    
    # LS Channel Estimate
    axes[0,0].plot(subcarriers, np.abs(results['channel_est_ls']), 'b-', linewidth=2)
    axes[0,0].set_title('LS Channel Estimate - Magnitude')
    axes[0,0].set_ylabel('Magnitude')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[1,0].plot(subcarriers, np.angle(results['channel_est_ls']), 'b-', linewidth=2)
    axes[1,0].set_title('LS Channel Estimate - Phase')
    axes[1,0].set_xlabel('Subcarrier Index')
    axes[1,0].set_ylabel('Phase (radians)')
    axes[1,0].grid(True, alpha=0.3)
    
    # MMSE Channel Estimate
    axes[0,1].plot(subcarriers, np.abs(results['channel_est_mmse']), 'r-', linewidth=2)
    axes[0,1].set_title('MMSE Channel Estimate - Magnitude')
    axes[0,1].set_ylabel('Magnitude')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,1].plot(subcarriers, np.angle(results['channel_est_mmse']), 'r-', linewidth=2)
    axes[1,1].set_title('MMSE Channel Estimate - Phase')
    axes[1,1].set_xlabel('Subcarrier Index')
    axes[1,1].set_ylabel('Phase (radians)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('channel_estimates_comparison.png', dpi=150, bbox_inches='tight')
        logging.info("Channel estimates comparison saved to 'channel_estimates_comparison.png'")
    
    # 2. Constellation Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data symbols for demodulation
    # Use correct pilot indices for IEEE 802.11 positions [-21, -7, 7, 21]
    pilot_indices_natural = np.where(params.pilot_pattern)[0]  # [11, 25, 39, 53]
    # After fftshift, these natural indices correspond to the correct centered positions
    pilot_indices_centered = pilot_indices_natural  # No shift needed!
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_centered] = False
    
    # ZF+LS Constellation
    data_symbols_zf = results['equalized_zf_ls'][data_mask]
    axes[0].scatter(np.real(data_symbols_zf), np.imag(data_symbols_zf), alpha=0.6, s=20)
    axes[0].set_title('ZF+LS Equalization')
    axes[0].set_xlabel('In-phase')
    axes[0].set_ylabel('Quadrature')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # MMSE+MMSE Constellation
    data_symbols_mmse = results['equalized_mmse_mmse'][data_mask]
    axes[1].scatter(np.real(data_symbols_mmse), np.imag(data_symbols_mmse), alpha=0.6, s=20)
    axes[1].set_title('MMSE+MMSE Equalization')
    axes[1].set_xlabel('In-phase')
    axes[1].set_ylabel('Quadrature')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('constellation_comparison.png', dpi=150, bbox_inches='tight')
        logging.info("Constellation comparison saved to 'constellation_comparison.png'")
    
    return fig

def main():
    """Main simulation function."""
    print("=" * 80)
    print("64-QAM OFDM RECEIVER SIMULATION")
    print("IEEE 802.11-class Baseband Receiver Design")
    print("=" * 80)
    
    # Simulation parameters
    snr_db = 15
    cfo_hz = 343.75e3  # ~1.1 subcarriers at 20 MHz BW
    timing_offset_samples = 43
    
    print(f"Simulation Parameters:")
    print(f"  SNR: {snr_db} dB")
    print(f"  CFO: {cfo_hz/1e3:.1f} kHz ({cfo_hz/params.subcarrier_spacing:.2f} subcarriers)")
    print(f"  Timing Offset: {timing_offset_samples} samples")
    print(f"  Channel: 5-tap Rayleigh fading")
    print()
    
    # Generate test signal
    logging.info("Generating test OFDM frame...")
    tx_frame, original_data_bits = generate_test_frame()
    
    # Apply channel impairments
    logging.info("Applying channel impairments...")
    rx_signal, actual_timing_offset = simulate_channel_impairments(
        tx_frame, snr_db=snr_db, cfo_hz=cfo_hz, timing_offset_samples=timing_offset_samples)
    
    # Process through receiver chain
    logging.info("Processing through receiver chain...")
    results = full_receiver_chain(rx_signal, actual_timing_offset, snr_db=snr_db)
    
    # Analyze performance
    logging.info("Analyzing performance...")
    performance = analyze_performance(results, original_data_bits)
    
    # Generate plots
    logging.info("Generating result plots...")
    plot_results(results, save_plots=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"Timing Synchronization:")
    print(f"  Estimated offset: {results['timing_offset_est']} samples")
    print(f"  Actual offset: {results['timing_offset_actual']} samples")
    print(f"  Error: {results['timing_error']} samples")
    
    print(f"\nCFO Estimation:")
    if 'cfo_estimates' in results and results['cfo_estimates']:
        cfo_est = results['cfo_estimates']
        print(f"  Estimated CFO: {cfo_est:.4f} subcarriers")
        print(f"  Actual CFO: {cfo_hz/params.subcarrier_spacing:.4f} subcarriers")
    
    print(f"\nBit Error Rate (BER):")
    for method, ber in performance.items():
        if 'BER' in method:
            if ber == float('inf'):
                print(f"  {method.replace('BER_', '')}: Failed")
            else:
                print(f"  {method.replace('BER_', '')}: {ber:.6f}")
    
    print("\nOutput Files Generated:")
    print("  - channel_estimates_comparison.png")
    print("  - constellation_comparison.png")
    print("  - rx_constellation_after_full_cfo.png (from CFO module)")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
