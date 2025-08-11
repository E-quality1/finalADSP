"""
BER Curve Analysis and ICI Analysis for OFDM Receiver

This module generates comprehensive BER vs SNR curves and analyzes
Inter-Carrier Interference (ICI) for the 64-QAM OFDM receiver.

Performance targets (IEEE 802.11-class):
- BER < 10^-3 at SNR = 15 dB for 64-QAM
- CFO tolerance: < 2% of subcarrier spacing
- Channel: 5-tap Rayleigh fading, 200ns RMS delay spread
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Import OFDM modules
from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp, generate_ofdm_frame
from ofdm.channel import apply_channel_effects
from ofdm.equalization import estimate_and_correct_cfo, zf_equalizer, mmse_equalizer
from ofdm.channel_estimation import estimate_channel_from_ofdm_symbol

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def simulate_single_frame(snr_db, cfo_hz=0, timing_offset=0, num_trials=1):
    """
    Simulate a single OFDM frame transmission and return BER results.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        cfo_hz: Carrier frequency offset in Hz
        timing_offset: Timing offset in samples
        num_trials: Number of independent trials to average
    
    Returns:
        results: Dictionary containing BER results for different methods
    """
    results = {
        'ber_zf_ls': [],
        'ber_mmse_mmse': [],
        'cfo_errors': [],
        'timing_errors': []
    }
    
    for trial in range(num_trials):
        try:
            # Generate test data
            num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
            original_data_bits = np.random.randint(0, 2, num_data_bits)
            
            # Generate OFDM frame
            tx_frame = generate_ofdm_frame()
            
            # Apply channel effects
            delay_s = timing_offset / params.sampling_rate
            rx_signal, actual_delay = apply_channel_effects(
                tx_frame, cfo=cfo_hz, snr_db=snr_db, delay_s=delay_s, apply_fading=True)
            
            # Receiver processing with perfect timing (for controlled testing)
            rx_symbol_freq_corrected, cfo_estimates = estimate_and_correct_cfo(rx_signal, actual_delay)
            
            # Channel estimation
            channel_est_ls, _ = estimate_channel_from_ofdm_symbol(rx_symbol_freq_corrected, method='ls')
            channel_est_mmse, _ = estimate_channel_from_ofdm_symbol(rx_symbol_freq_corrected, method='mmse', snr_db=snr_db)
            
            # Equalization
            equalized_zf_ls = zf_equalizer(rx_symbol_freq_corrected, channel_est_ls)
            noise_var = 10**(-snr_db/10)
            equalized_mmse_mmse = mmse_equalizer(rx_symbol_freq_corrected, channel_est_mmse, noise_var)
            
            # Extract data subcarriers
            # Use correct pilot indices for IEEE 802.11 positions [-21, -7, 7, 21]
            pilot_indices_natural = np.where(params.pilot_pattern)[0]  # [11, 25, 39, 53]
            # After fftshift, these natural indices correspond to the correct centered positions
            pilot_indices_centered = pilot_indices_natural  # No shift needed!
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_centered] = False
            
            # Calculate BER for each method
            methods = {
                'zf_ls': equalized_zf_ls,
                'mmse_mmse': equalized_mmse_mmse
            }
            
            for method_name, equalized_symbols in methods.items():
                data_symbols_eq = equalized_symbols[data_mask]
                
                try:
                    demod_bits = qam_demod(data_symbols_eq)
                    if len(demod_bits) == len(original_data_bits):
                        bit_errors = np.sum(demod_bits != original_data_bits)
                        ber = bit_errors / len(original_data_bits)
                        results[f'ber_{method_name}'].append(ber)
                    else:
                        results[f'ber_{method_name}'].append(1.0)  # Complete failure
                except:
                    results[f'ber_{method_name}'].append(1.0)  # Demodulation failed
            
            # Store CFO and timing errors
            actual_cfo = cfo_hz / params.subcarrier_spacing
            results['cfo_errors'].append(abs(cfo_estimates - actual_cfo) if cfo_estimates else float('inf'))
            results['timing_errors'].append(abs(actual_delay - timing_offset))
            
        except Exception as e:
            # Handle any simulation errors
            results['ber_zf_ls'].append(1.0)
            results['ber_mmse_mmse'].append(1.0)
            results['cfo_errors'].append(float('inf'))
            results['timing_errors'].append(float('inf'))
    
    # Average results across trials
    averaged_results = {}
    for key, values in results.items():
        if values:
            averaged_results[key] = np.mean(values)
        else:
            averaged_results[key] = 1.0 if 'ber' in key else float('inf')
    
    return averaged_results

def generate_ber_curves(snr_range=None, num_trials=10, save_results=True):
    """
    Generate BER vs SNR curves for different equalization methods.
    
    Args:
        snr_range: Range of SNR values to test (dB)
        num_trials: Number of trials per SNR point
        save_results: Whether to save results to file
    
    Returns:
        results: Dictionary containing BER curves and metadata
    """
    if snr_range is None:
        snr_range = np.arange(0, 21, 2)  # 0 to 20 dB in 2 dB steps
    
    print("=" * 60)
    print("GENERATING BER CURVES")
    print("=" * 60)
    print(f"SNR range: {snr_range[0]} to {snr_range[-1]} dB")
    print(f"Trials per SNR: {num_trials}")
    print(f"Total simulations: {len(snr_range) * num_trials}")
    print()
    
    results = {
        'snr_db': snr_range,
        'ber_zf_ls': [],
        'ber_mmse_mmse': [],
        'cfo_errors': [],
        'timing_errors': []
    }
    
    # Run simulations for each SNR point
    for snr_db in tqdm(snr_range, desc="SNR Points"):
        logging.info(f"Simulating SNR = {snr_db} dB")
        
        # Simulate multiple trials for this SNR
        trial_results = simulate_single_frame(snr_db, cfo_hz=343.75e3, timing_offset=43, num_trials=num_trials)
        
        # Store averaged results
        results['ber_zf_ls'].append(trial_results['ber_zf_ls'])
        results['ber_mmse_mmse'].append(trial_results['ber_mmse_mmse'])
        results['cfo_errors'].append(trial_results['cfo_errors'])
        results['timing_errors'].append(trial_results['timing_errors'])
        
        logging.info(f"  ZF+LS BER: {trial_results['ber_zf_ls']:.6f}")
        logging.info(f"  MMSE+MMSE BER: {trial_results['ber_mmse_mmse']:.6f}")
    
    # Convert to numpy arrays
    for key in ['ber_zf_ls', 'ber_mmse_mmse', 'cfo_errors', 'timing_errors']:
        results[key] = np.array(results[key])
    
    # Save results if requested
    if save_results:
        np.savez('ber_curves_results.npz', **results)
        logging.info("BER curve results saved to 'ber_curves_results.npz'")
    
    return results

def plot_ber_curves(results, save_path='ber_curves.png'):
    """
    Plot BER vs SNR curves.
    
    Args:
        results: Results dictionary from generate_ber_curves()
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot BER curves
    ax.semilogy(results['snr_db'], results['ber_zf_ls'], 'b-o', 
                linewidth=2, markersize=6, label='ZF + LS Channel Est.')
    ax.semilogy(results['snr_db'], results['ber_mmse_mmse'], 'r-s', 
                linewidth=2, markersize=6, label='MMSE + MMSE Channel Est.')
    
    # Add performance target line
    target_ber = 1e-3
    target_snr = 15
    ax.axhline(y=target_ber, color='k', linestyle='--', alpha=0.7, 
               label=f'Target BER = {target_ber:.0e}')
    ax.axvline(x=target_snr, color='k', linestyle='--', alpha=0.7, 
               label=f'Target SNR = {target_snr} dB')
    
    # Formatting
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('64-QAM OFDM Receiver BER Performance\n(5-tap Rayleigh Channel, CFO = 1.1 subcarriers)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([1e-6, 1])
    ax.set_xlim([results['snr_db'][0], results['snr_db'][-1]])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"BER curves saved to '{save_path}'")
    
    return fig

def analyze_ici_effects(cfo_range=None, snr_db=15, num_trials=5):
    """
    Analyze Inter-Carrier Interference (ICI) effects due to CFO.
    
    Args:
        cfo_range: Range of CFO values as fraction of subcarrier spacing
        snr_db: Fixed SNR for ICI analysis
        num_trials: Number of trials per CFO point
    
    Returns:
        ici_results: Dictionary containing ICI analysis results
    """
    if cfo_range is None:
        cfo_range = np.linspace(0, 0.5, 11)  # 0 to 50% of subcarrier spacing
    
    print("\n" + "=" * 60)
    print("ANALYZING ICI EFFECTS")
    print("=" * 60)
    print(f"CFO range: {cfo_range[0]:.2f} to {cfo_range[-1]:.2f} (fraction of subcarrier spacing)")
    print(f"Fixed SNR: {snr_db} dB")
    print(f"Trials per CFO: {num_trials}")
    print()
    
    ici_results = {
        'cfo_normalized': cfo_range,
        'ber_zf_ls': [],
        'ber_mmse_mmse': [],
        'sinr_degradation': []  # Signal-to-Interference-plus-Noise Ratio degradation
    }
    
    # Reference BER with no CFO
    ref_results = simulate_single_frame(snr_db, cfo_hz=0, timing_offset=0, num_trials=num_trials)
    ref_ber_zf = ref_results['ber_zf_ls']
    ref_ber_mmse = ref_results['ber_mmse_mmse']
    
    for cfo_norm in tqdm(cfo_range, desc="CFO Points"):
        cfo_hz = cfo_norm * params.subcarrier_spacing
        
        # Simulate with CFO
        trial_results = simulate_single_frame(snr_db, cfo_hz=cfo_hz, timing_offset=0, num_trials=num_trials)
        
        ici_results['ber_zf_ls'].append(trial_results['ber_zf_ls'])
        ici_results['ber_mmse_mmse'].append(trial_results['ber_mmse_mmse'])
        
        # Calculate SINR degradation (approximation)
        if ref_ber_zf > 0 and trial_results['ber_zf_ls'] > 0:
            sinr_degradation = 10 * np.log10(trial_results['ber_zf_ls'] / ref_ber_zf)
        else:
            sinr_degradation = 0
        ici_results['sinr_degradation'].append(sinr_degradation)
        
        logging.info(f"CFO = {cfo_norm:.2f}: ZF BER = {trial_results['ber_zf_ls']:.6f}, "
                    f"MMSE BER = {trial_results['ber_mmse_mmse']:.6f}")
    
    # Convert to numpy arrays
    for key in ['ber_zf_ls', 'ber_mmse_mmse', 'sinr_degradation']:
        ici_results[key] = np.array(ici_results[key])
    
    return ici_results

def plot_ici_analysis(ici_results, save_path='ici_analysis.png'):
    """
    Plot ICI analysis results.
    
    Args:
        ici_results: Results from analyze_ici_effects()
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: BER vs CFO
    ax1.semilogy(ici_results['cfo_normalized'], ici_results['ber_zf_ls'], 'b-o', 
                linewidth=2, markersize=6, label='ZF + LS')
    ax1.semilogy(ici_results['cfo_normalized'], ici_results['ber_mmse_mmse'], 'r-s', 
                linewidth=2, markersize=6, label='MMSE + MMSE')
    
    # Add CFO tolerance line (2% of subcarrier spacing)
    cfo_tolerance = 0.02
    ax1.axvline(x=cfo_tolerance, color='k', linestyle='--', alpha=0.7, 
               label=f'CFO Tolerance = {cfo_tolerance:.0%}')
    
    ax1.set_xlabel('CFO (fraction of subcarrier spacing)', fontsize=12)
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax1.set_title('ICI Effects: BER vs Carrier Frequency Offset', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([1e-6, 1])
    
    # Plot 2: SINR Degradation vs CFO
    ax2.plot(ici_results['cfo_normalized'], ici_results['sinr_degradation'], 'g-^', 
             linewidth=2, markersize=6, label='SINR Degradation')
    ax2.axvline(x=cfo_tolerance, color='k', linestyle='--', alpha=0.7, 
               label=f'CFO Tolerance = {cfo_tolerance:.0%}')
    
    ax2.set_xlabel('CFO (fraction of subcarrier spacing)', fontsize=12)
    ax2.set_ylabel('SINR Degradation (dB)', fontsize=12)
    ax2.set_title('ICI Effects: SINR Degradation vs CFO', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"ICI analysis plot saved to '{save_path}'")
    
    return fig

def main():
    """Main function to run BER curve generation and ICI analysis."""
    print("=" * 80)
    print("OFDM RECEIVER BER CURVE ANALYSIS AND ICI STUDY")
    print("=" * 80)
    
    # Generate BER curves
    logging.info("Starting BER curve generation...")
    ber_results = generate_ber_curves(snr_range=np.arange(5, 21, 2), num_trials=20)
    
    # Plot BER curves
    logging.info("Plotting BER curves...")
    plot_ber_curves(ber_results)
    
    # Analyze ICI effects
    logging.info("Starting ICI analysis...")
    ici_results = analyze_ici_effects(cfo_range=np.linspace(0, 0.1, 11), snr_db=15, num_trials=10)
    
    # Plot ICI analysis
    logging.info("Plotting ICI analysis...")
    plot_ici_analysis(ici_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BER ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Find BER at target SNR (15 dB)
    target_snr = 15
    if target_snr in ber_results['snr_db']:
        idx = np.where(ber_results['snr_db'] == target_snr)[0][0]
        ber_zf_at_target = ber_results['ber_zf_ls'][idx]
        ber_mmse_at_target = ber_results['ber_mmse_mmse'][idx]
        
        print(f"Performance at {target_snr} dB SNR:")
        print(f"  ZF + LS:     BER = {ber_zf_at_target:.2e}")
        print(f"  MMSE + MMSE: BER = {ber_mmse_at_target:.2e}")
        print(f"  Target:      BER < 1.0e-03")
        
        target_met_zf = ber_zf_at_target < 1e-3
        target_met_mmse = ber_mmse_at_target < 1e-3
        print(f"  ZF Target Met:   {'✓ YES' if target_met_zf else '✗ NO'}")
        print(f"  MMSE Target Met: {'✓ YES' if target_met_mmse else '✗ NO'}")
    
    # ICI tolerance analysis
    cfo_tolerance = 0.02  # 2% of subcarrier spacing
    if cfo_tolerance in ici_results['cfo_normalized']:
        idx = np.where(np.isclose(ici_results['cfo_normalized'], cfo_tolerance))[0]
        if len(idx) > 0:
            ber_at_tolerance = ici_results['ber_zf_ls'][idx[0]]
            print(f"\nICI Analysis at {cfo_tolerance:.0%} CFO tolerance:")
            print(f"  BER with CFO: {ber_at_tolerance:.2e}")
            print(f"  Tolerance Met: {'✓ YES' if ber_at_tolerance < 1e-2 else '✗ NO'}")
    
    print("\nOutput Files Generated:")
    print("  - ber_curves.png")
    print("  - ici_analysis.png")
    print("  - ber_curves_results.npz")
    
    print("\n" + "=" * 80)
    print("BER ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
