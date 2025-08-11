"""
Generate Constellation Diagrams

This script creates constellation diagrams showing 64-QAM performance
before and after equalization at different SNR levels.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib

# Force reload of modules
if 'ofdm.transmitter' in sys.modules:
    importlib.reload(sys.modules['ofdm.transmitter'])
if 'ofdm.system' in sys.modules:
    importlib.reload(sys.modules['ofdm.system'])

from ofdm.system import params
from ofdm.transmitter import qam_mod, insert_pilots, ifft_with_cp
from ofdm.channel import add_awgn, apply_channel_effects
from ofdm.equalization import estimate_and_correct_cfo
from ofdm.channel_estimation import ls_channel_estimation
from ofdm.equalization import zf_equalizer

def generate_64qam_reference():
    """Generate ideal 64-QAM constellation points."""
    # 64-QAM constellation (8x8 grid)
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    constellation = []
    
    for i in levels:
        for q in levels:
            constellation.append(complex(i, q))
    
    return np.array(constellation)

def create_constellation_diagrams():
    """Create constellation diagrams showing receiver performance."""
    print("Generating constellation diagrams...")
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Transmitter
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Test at different SNR levels
    snr_levels = [10, 20, 30]  # dB
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, snr_db in enumerate(snr_levels):
        print(f"Processing SNR = {snr_db} dB...")
        
        # Add channel effects
        rx_signal = add_awgn(time_signal, snr_db=snr_db)
        
        # Perfect timing (for constellation clarity)
        rx_time_no_cp = rx_signal[params.CP:params.CP + params.N]
        rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
        rx_freq_centered = np.fft.fftshift(rx_freq_natural)
        
        # Extract data symbols before equalization
        pilot_indices_natural = np.where(params.pilot_pattern)[0]
        pilot_indices_centered = pilot_indices_natural
        data_mask = np.ones(params.N, dtype=bool)
        data_mask[pilot_indices_centered] = False
        rx_data_before_eq = rx_freq_centered[data_mask]
        
        # Apply CFO correction and equalization
        rx_freq_cfo_corrected = estimate_and_correct_cfo(rx_freq_centered)
        channel_est = ls_channel_estimation(rx_freq_cfo_corrected)
        rx_freq_equalized = zf_equalizer(rx_freq_cfo_corrected, channel_est)
        rx_data_after_eq = rx_freq_equalized[data_mask]
        
        # Plot before equalization (top row)
        ax_before = axes[0, idx]
        ax_before.scatter(rx_data_before_eq.real, rx_data_before_eq.imag, 
                         alpha=0.6, s=20, c='red', label='Received')
        
        # Add ideal constellation for reference
        ideal_constellation = generate_64qam_reference()
        ax_before.scatter(ideal_constellation.real, ideal_constellation.imag, 
                         alpha=0.8, s=15, c='blue', marker='x', label='Ideal')
        
        ax_before.set_title(f'Before Equalization\nSNR = {snr_db} dB', fontsize=12)
        ax_before.set_xlabel('In-phase (I)')
        ax_before.set_ylabel('Quadrature (Q)')
        ax_before.grid(True, alpha=0.3)
        ax_before.legend()
        ax_before.set_xlim(-10, 10)
        ax_before.set_ylim(-10, 10)
        
        # Plot after equalization (bottom row)
        ax_after = axes[1, idx]
        ax_after.scatter(rx_data_after_eq.real, rx_data_after_eq.imag, 
                        alpha=0.6, s=20, c='green', label='Equalized')
        
        # Add ideal constellation for reference
        ax_after.scatter(ideal_constellation.real, ideal_constellation.imag, 
                        alpha=0.8, s=15, c='blue', marker='x', label='Ideal')
        
        ax_after.set_title(f'After Equalization\nSNR = {snr_db} dB', fontsize=12)
        ax_after.set_xlabel('In-phase (I)')
        ax_after.set_ylabel('Quadrature (Q)')
        ax_after.grid(True, alpha=0.3)
        ax_after.legend()
        ax_after.set_xlim(-10, 10)
        ax_after.set_ylim(-10, 10)
    
    plt.suptitle('64-QAM Constellation Diagrams: Receiver Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('constellation_diagrams.png', dpi=300, bbox_inches='tight')
    plt.savefig('constellation_diagrams.pdf', bbox_inches='tight')
    print("Constellation diagrams saved to 'constellation_diagrams.png' and '.pdf'")
    
    return fig

def create_single_constellation_comparison():
    """Create a focused constellation comparison at one SNR level."""
    print("Generating focused constellation comparison...")
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Transmitter
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Test at 15 dB SNR (moderate performance)
    snr_db = 15
    rx_signal = add_awgn(time_signal, snr_db=snr_db)
    
    # Perfect timing
    rx_time_no_cp = rx_signal[params.CP:params.CP + params.N]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Extract data symbols
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_centered = pilot_indices_natural
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_centered] = False
    rx_data_before_eq = rx_freq_centered[data_mask]
    
    # Apply equalization
    rx_freq_cfo_corrected = estimate_and_correct_cfo(rx_freq_centered)
    channel_est = ls_channel_estimation(rx_freq_cfo_corrected)
    rx_freq_equalized = zf_equalizer(rx_freq_cfo_corrected, channel_est)
    rx_data_after_eq = rx_freq_equalized[data_mask]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ideal constellation
    ideal_constellation = generate_64qam_reference()
    
    # Before equalization
    ax1.scatter(rx_data_before_eq.real, rx_data_before_eq.imag, 
               alpha=0.6, s=25, c='red', label='Received Symbols')
    ax1.scatter(ideal_constellation.real, ideal_constellation.imag, 
               alpha=0.8, s=20, c='blue', marker='x', label='Ideal 64-QAM')
    ax1.set_title(f'Before Equalization (SNR = {snr_db} dB)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('In-phase (I)')
    ax1.set_ylabel('Quadrature (Q)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    
    # After equalization
    ax2.scatter(rx_data_after_eq.real, rx_data_after_eq.imag, 
               alpha=0.6, s=25, c='green', label='Equalized Symbols')
    ax2.scatter(ideal_constellation.real, ideal_constellation.imag, 
               alpha=0.8, s=20, c='blue', marker='x', label='Ideal 64-QAM')
    ax2.set_title(f'After Equalization (SNR = {snr_db} dB)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('In-phase (I)')
    ax2.set_ylabel('Quadrature (Q)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    
    plt.suptitle('64-QAM Constellation: Impact of Receiver Equalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('constellation_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('constellation_comparison.pdf', bbox_inches='tight')
    print("Constellation comparison saved to 'constellation_comparison.png' and '.pdf'")
    
    return fig

def main():
    """Generate constellation diagrams."""
    print("CONSTELLATION DIAGRAM GENERATION")
    print("="*50)
    
    # Generate comprehensive constellation diagrams
    fig1 = create_constellation_diagrams()
    
    # Generate focused comparison
    fig2 = create_single_constellation_comparison()
    
    print("\n" + "="*50)
    print("CONSTELLATION DIAGRAMS COMPLETE")
    print("="*50)
    print("Generated files:")
    print("  - constellation_diagrams.png (comprehensive view)")
    print("  - constellation_comparison.png (focused comparison)")
    print("  - PDF versions also available")

if __name__ == "__main__":
    main()
