"""
Generate Simple Constellation Diagrams

This script creates constellation diagrams showing 64-QAM performance
with a simplified approach to avoid API complexities.
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
from ofdm.channel import add_awgn

def generate_64qam_reference():
    """Generate ideal 64-QAM constellation points."""
    # 64-QAM constellation (8x8 grid)
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    constellation = []
    
    for i in levels:
        for q in levels:
            constellation.append(complex(i, q))
    
    return np.array(constellation)

def create_constellation_comparison():
    """Create constellation comparison showing noise effects."""
    print("Generating constellation comparison...")
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Transmitter
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Create comparison at different SNR levels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Ideal constellation
    ideal_constellation = generate_64qam_reference()
    
    # Test conditions
    conditions = [
        {'snr': None, 'title': 'Ideal 64-QAM\n(No Noise)', 'ax': axes[0,0]},
        {'snr': 10, 'title': 'Low SNR (10 dB)\nWith AWGN', 'ax': axes[0,1]},
        {'snr': 20, 'title': 'Medium SNR (20 dB)\nWith AWGN', 'ax': axes[1,0]},
        {'snr': 30, 'title': 'High SNR (30 dB)\nWith AWGN', 'ax': axes[1,1]},
    ]
    
    for condition in conditions:
        ax = condition['ax']
        
        if condition['snr'] is None:
            # Ideal case - just show the original symbols
            symbols_to_plot = original_symbols
            color = 'blue'
            alpha = 0.8
        else:
            # Add noise and extract symbols
            rx_signal = add_awgn(time_signal, snr_db=condition['snr'])
            
            # Simple receiver (perfect timing, no CFO/channel effects)
            rx_time_no_cp = rx_signal[params.CP:params.CP + params.N]
            rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
            rx_freq_centered = np.fft.fftshift(rx_freq_natural)
            
            # Extract data symbols
            pilot_indices_natural = np.where(params.pilot_pattern)[0]
            pilot_indices_centered = pilot_indices_natural
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_centered] = False
            symbols_to_plot = rx_freq_centered[data_mask]
            
            color = 'red'
            alpha = 0.6
        
        # Plot symbols
        ax.scatter(symbols_to_plot.real, symbols_to_plot.imag, 
                  alpha=alpha, s=25, c=color, label='Received')
        
        # Add ideal constellation reference (except for ideal case)
        if condition['snr'] is not None:
            ax.scatter(ideal_constellation.real, ideal_constellation.imag, 
                      alpha=0.3, s=15, c='blue', marker='x', label='Ideal')
            ax.legend()
        
        ax.set_title(condition['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    
    plt.suptitle('64-QAM Constellation Diagrams: Effect of AWGN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('constellation_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('constellation_comparison.pdf', bbox_inches='tight')
    print("Constellation comparison saved to 'constellation_comparison.png' and '.pdf'")
    
    return fig

def create_ideal_64qam_constellation():
    """Create a clean ideal 64-QAM constellation diagram."""
    print("Generating ideal 64-QAM constellation...")
    
    # Generate ideal constellation
    ideal_constellation = generate_64qam_reference()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot constellation points
    ax.scatter(ideal_constellation.real, ideal_constellation.imag, 
              s=100, c='blue', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add bit labels for some points (optional)
    # This would require mapping symbols to bits, which is complex for 64-QAM
    
    ax.set_title('Ideal 64-QAM Constellation', fontsize=16, fontweight='bold')
    ax.set_xlabel('In-phase (I)', fontsize=12)
    ax.set_ylabel('Quadrature (Q)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.set_aspect('equal')
    
    # Add constellation info
    textstr = f'64-QAM Constellation\n64 symbols (8×8 grid)\n6 bits per symbol'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('ideal_64qam_constellation.png', dpi=300, bbox_inches='tight')
    plt.savefig('ideal_64qam_constellation.pdf', bbox_inches='tight')
    print("Ideal 64-QAM constellation saved to 'ideal_64qam_constellation.png' and '.pdf'")
    
    return fig

def main():
    """Generate constellation diagrams."""
    print("CONSTELLATION DIAGRAM GENERATION")
    print("="*50)
    
    # Generate ideal constellation
    fig1 = create_ideal_64qam_constellation()
    
    # Generate comparison with noise effects
    fig2 = create_constellation_comparison()
    
    print("\n" + "="*50)
    print("CONSTELLATION DIAGRAMS COMPLETE")
    print("="*50)
    print("Generated files:")
    print("  - ideal_64qam_constellation.png (clean reference)")
    print("  - constellation_comparison.png (noise effects)")
    print("  - PDF versions also available")

if __name__ == "__main__":
    main()
