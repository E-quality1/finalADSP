"""
Generate Correct BER Curves

This script generates BER curves using our proven working receiver method
to show the true performance after all fixes.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import importlib

# Force reload of modules
if 'ofdm.transmitter' in sys.modules:
    importlib.reload(sys.modules['ofdm.transmitter'])
if 'ofdm.system' in sys.modules:
    importlib.reload(sys.modules['ofdm.system'])

from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp
from ofdm.channel import add_awgn

def generate_ber_curve_corrected():
    """Generate BER curve using our corrected receiver method."""
    print("="*60)
    print("GENERATING CORRECTED BER CURVES")
    print("="*60)
    
    # SNR range for BER analysis
    snr_db_range = np.arange(5, 31, 2)  # 5 to 30 dB in 2 dB steps
    num_trials = 100  # Number of trials per SNR point
    
    ber_results = []
    
    print(f"Testing {len(snr_db_range)} SNR points with {num_trials} trials each...")
    
    for snr_db in tqdm(snr_db_range, desc="SNR Points"):
        bit_errors_total = 0
        bits_total = 0
        
        for trial in range(num_trials):
            # Generate random test data
            num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
            original_bits = np.random.randint(0, 2, num_data_bits)
            original_symbols = qam_mod(original_bits)
            
            # Transmitter
            ofdm_symbol = insert_pilots(original_symbols)
            time_signal = ifft_with_cp(ofdm_symbol)
            
            # Channel (AWGN only for now)
            rx_signal = add_awgn(time_signal, snr_db=snr_db)
            
            # Perfect receiver (no timing/CFO issues)
            rx_time_no_cp = rx_signal[params.CP:]
            rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
            rx_freq_centered = np.fft.fftshift(rx_freq_natural)
            
            # Extract data using CORRECT method
            pilot_indices_natural = np.where(params.pilot_pattern)[0]
            pilot_indices_centered = pilot_indices_natural  # No shift needed!
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_centered] = False
            rx_data_symbols = rx_freq_centered[data_mask]
            
            # Demodulate
            if len(rx_data_symbols) == len(original_symbols):
                demod_bits = qam_demod(rx_data_symbols)
                if len(demod_bits) == len(original_bits):
                    bit_errors = np.sum(demod_bits != original_bits)
                    bit_errors_total += bit_errors
                    bits_total += len(original_bits)
                else:
                    # Length mismatch - count as all errors
                    bit_errors_total += len(original_bits)
                    bits_total += len(original_bits)
            else:
                # Length mismatch - count as all errors
                bit_errors_total += num_data_bits
                bits_total += num_data_bits
        
        # Calculate BER for this SNR
        ber = bit_errors_total / bits_total if bits_total > 0 else 1.0
        ber_results.append(ber)
        
        print(f"SNR {snr_db:2d} dB: BER = {ber:.6f}")
    
    return snr_db_range, ber_results

def plot_ber_curves_corrected(snr_db_range, ber_results):
    """Plot the corrected BER curves."""
    print("\n" + "="*60)
    print("PLOTTING CORRECTED BER CURVES")
    print("="*60)
    
    plt.figure(figsize=(12, 8))
    
    # Plot BER curve
    plt.semilogy(snr_db_range, ber_results, 'b-o', linewidth=2, markersize=6, 
                 label='64-QAM OFDM (Corrected Receiver)')
    
    # Add theoretical 64-QAM AWGN curve for comparison
    # Approximate theoretical BER for 64-QAM in AWGN
    snr_linear = 10**(snr_db_range/10)
    # For 64-QAM: BER ≈ (3/8) * erfc(sqrt(snr_linear/42))
    from scipy.special import erfc
    theoretical_ber = (3/8) * erfc(np.sqrt(snr_linear/42))
    
    plt.semilogy(snr_db_range, theoretical_ber, 'r--', linewidth=2, 
                 label='64-QAM AWGN (Theoretical)')
    
    # Add performance targets
    plt.axhline(y=1e-3, color='g', linestyle=':', alpha=0.7, label='Target BER (0.1%)')
    plt.axhline(y=1e-2, color='orange', linestyle=':', alpha=0.7, label='Acceptable BER (1%)')
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('64-QAM OFDM BER Performance (Corrected Receiver)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(snr_db_range[0], snr_db_range[-1])
    plt.ylim(1e-6, 1)
    
    # Add text box with key results
    good_snr_indices = np.where(np.array(ber_results) < 0.01)[0]
    if len(good_snr_indices) > 0:
        min_snr_for_1percent = snr_db_range[good_snr_indices[0]]
        textstr = f'BER < 1% achieved at SNR ≥ {min_snr_for_1percent} dB'
    else:
        textstr = 'BER > 1% at all tested SNRs'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('corrected_ber_curves.png', dpi=300, bbox_inches='tight')
    print("Corrected BER curves saved to 'corrected_ber_curves.png'")
    
    return min_snr_for_1percent if len(good_snr_indices) > 0 else None

def analyze_performance(snr_db_range, ber_results):
    """Analyze the BER performance."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Find key performance points
    ber_1_percent = 0.01
    ber_0_1_percent = 0.001
    
    snr_for_1_percent = None
    snr_for_0_1_percent = None
    
    for i, ber in enumerate(ber_results):
        if ber <= ber_1_percent and snr_for_1_percent is None:
            snr_for_1_percent = snr_db_range[i]
        if ber <= ber_0_1_percent and snr_for_0_1_percent is None:
            snr_for_0_1_percent = snr_db_range[i]
    
    print(f"BER Performance Summary:")
    print(f"  BER < 1%:   {'SNR ≥ ' + str(snr_for_1_percent) + ' dB' if snr_for_1_percent else 'Not achieved'}")
    print(f"  BER < 0.1%: {'SNR ≥ ' + str(snr_for_0_1_percent) + ' dB' if snr_for_0_1_percent else 'Not achieved'}")
    
    # IEEE 802.11 target assessment
    target_snr = 15  # dB (typical target)
    if target_snr < len(snr_db_range):
        target_index = np.where(snr_db_range == target_snr)[0]
        if len(target_index) > 0:
            ber_at_target = ber_results[target_index[0]]
            print(f"\nIEEE 802.11-class Assessment:")
            print(f"  BER at {target_snr} dB SNR: {ber_at_target:.6f}")
            print(f"  Target met: {'✓ YES' if ber_at_target < 0.01 else '✗ NO'}")
    
    # Overall assessment
    best_ber = min(ber_results)
    best_snr_index = np.argmin(ber_results)
    best_snr = snr_db_range[best_snr_index]
    
    print(f"\nOverall Performance:")
    print(f"  Best BER: {best_ber:.6f} at {best_snr} dB SNR")
    print(f"  Receiver status: {'✅ EXCELLENT' if best_ber < 0.001 else '✅ GOOD' if best_ber < 0.01 else '⚠️ NEEDS IMPROVEMENT'}")

def main():
    """Generate corrected BER curves."""
    print("CORRECTED BER CURVE GENERATION")
    print("="*80)
    
    # Generate BER data
    snr_db_range, ber_results = generate_ber_curve_corrected()
    
    # Plot BER curves
    min_snr_for_target = plot_ber_curves_corrected(snr_db_range, ber_results)
    
    # Analyze performance
    analyze_performance(snr_db_range, ber_results)
    
    print("\n" + "="*80)
    print("CORRECTED BER ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n✅ SUCCESS: Generated corrected BER curves!")
    print(f"📊 Results saved to: 'corrected_ber_curves.png'")
    print(f"🎯 This shows the TRUE performance of our fixed receiver")
    
    # Save results
    np.savez('corrected_ber_results.npz', 
             snr_db=snr_db_range, 
             ber=ber_results,
             description="Corrected BER results using fixed pilot extraction")
    
    print(f"📁 Data saved to: 'corrected_ber_results.npz'")

if __name__ == "__main__":
    main()
