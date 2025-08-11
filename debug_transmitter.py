"""
Transmitter Diagnostic

This script investigates the transmitter pilot insertion and IFFT logic
to identify the root cause of incorrect pilot extraction in the receiver.
"""
import numpy as np
import matplotlib.pyplot as plt
from ofdm.system import params
from ofdm.transmitter import qam_mod, insert_pilots

def analyze_pilot_insertion():
    """Analyze pilot insertion in detail."""
    print("="*60)
    print("PILOT INSERTION ANALYSIS")
    print("="*60)
    
    # Create test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    test_bits = np.random.randint(0, 2, num_data_bits)
    data_symbols = qam_mod(test_bits)
    
    print(f"Data symbols: {len(data_symbols)}")
    print(f"Pilot pattern: {params.pilot_pattern}")
    print(f"Number of pilots: {np.sum(params.pilot_pattern)}")
    
    # Insert pilots
    ofdm_symbol = insert_pilots(data_symbols)
    print(f"OFDM symbol length: {len(ofdm_symbol)}")
    
    # Check pilot positions
    pilot_indices = np.where(params.pilot_pattern)[0]
    data_indices = np.where(~params.pilot_pattern)[0]
    
    print(f"Pilot indices (natural order): {pilot_indices}")
    print(f"Data indices (first 5): {data_indices[:5]}")
    
    # Verify pilots
    pilot_values = ofdm_symbol[pilot_indices]
    print(f"Pilot values: {pilot_values}")
    print(f"Expected pilot value: {params.pilot_value}")
    print(f"Pilots correct: {'✓' if np.allclose(pilot_values, params.pilot_value) else '✗'}")
    
    return ofdm_symbol, pilot_indices, data_indices

def analyze_ifft_shift_issue(ofdm_symbol, pilot_indices, data_indices):
    """Analyze the IFFT shift issue."""
    print("\n" + "="*60)
    print("IFFT SHIFT ANALYSIS")
    print("="*60)
    
    print("Current transmitter approach:")
    print("1. insert_pilots() creates symbol in natural order")
    print("2. ifft_with_cp() applies ifftshift then IFFT")
    
    # Show what happens with current approach
    print("\nCurrent approach:")
    ofdm_shifted_current = np.fft.ifftshift(ofdm_symbol)
    print(f"After ifftshift: pilot positions change")
    
    # Find where pilots end up after ifftshift
    pilot_positions_after_ifftshift = (pilot_indices + params.N//2) % params.N
    print(f"Pilots move from {pilot_indices} to {pilot_positions_after_ifftshift}")
    
    # Check pilot values after shift
    pilots_after_shift = ofdm_shifted_current[pilot_positions_after_ifftshift]
    print(f"Pilot values after ifftshift: {pilots_after_shift}")
    print(f"Still correct: {'✓' if np.allclose(pilots_after_shift, params.pilot_value) else '✗'}")
    
    # The correct approach should be:
    print("\nCorrect approach should be:")
    print("1. insert_pilots() creates symbol in centered order (DC at center)")
    print("2. ifft_with_cp() applies ifftshift then IFFT")
    print("   OR")
    print("1. insert_pilots() creates symbol in natural order")
    print("2. ifft_with_cp() applies IFFT directly (no ifftshift)")
    
    return ofdm_shifted_current

def test_both_approaches():
    """Test both transmitter approaches."""
    print("\n" + "="*60)
    print("COMPARING TRANSMITTER APPROACHES")
    print("="*60)
    
    # Create test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    test_bits = np.random.randint(0, 2, num_data_bits)
    data_symbols = qam_mod(test_bits)
    
    # Approach 1: Current (natural order + ifftshift)
    ofdm_symbol_natural = insert_pilots(data_symbols)
    ofdm_shifted = np.fft.ifftshift(ofdm_symbol_natural)
    time_signal_1 = np.fft.ifft(ofdm_shifted) * np.sqrt(params.N)
    
    # Approach 2: Fixed (natural order, no ifftshift)
    time_signal_2 = np.fft.ifft(ofdm_symbol_natural) * np.sqrt(params.N)
    
    # Approach 3: Alternative (centered order + ifftshift)
    # First create centered version of OFDM symbol
    ofdm_symbol_centered = np.fft.fftshift(ofdm_symbol_natural)
    ofdm_shifted_alt = np.fft.ifftshift(ofdm_symbol_centered)
    time_signal_3 = np.fft.ifft(ofdm_shifted_alt) * np.sqrt(params.N)
    
    print(f"Approach 1 (current): natural + ifftshift")
    print(f"Approach 2 (fixed): natural, no ifftshift")  
    print(f"Approach 3 (alternative): centered + ifftshift")
    
    # Test receiver processing for each approach
    approaches = [
        ("Current (natural + ifftshift)", time_signal_1),
        ("Fixed (natural, no ifftshift)", time_signal_2),
        ("Alternative (centered + ifftshift)", time_signal_3)
    ]
    
    for name, time_signal in approaches:
        print(f"\n{name}:")
        
        # Receiver: FFT + fftshift
        rx_freq = np.fft.fft(time_signal) / np.sqrt(params.N)
        rx_freq_centered = np.fft.fftshift(rx_freq)
        
        # Extract pilots using correct indices for centered spectrum
        pilot_indices_natural = np.where(params.pilot_pattern)[0]
        pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
        
        rx_pilots = rx_freq_centered[pilot_indices_after_fftshift]
        
        print(f"  Extracted pilots: {rx_pilots}")
        print(f"  Expected pilots: {[params.pilot_value]*len(pilot_indices_natural)}")
        pilots_ok = np.allclose(rx_pilots, params.pilot_value, rtol=0.1)
        print(f"  Pilots correct: {'✓' if pilots_ok else '✗'}")
        
        # Extract data and test BER
        data_mask = np.ones(params.N, dtype=bool)
        data_mask[pilot_indices_after_fftshift] = False
        rx_data = rx_freq_centered[data_mask]
        
        if len(rx_data) == len(data_symbols):
            symbol_errors = np.sum(~np.isclose(rx_data, data_symbols, rtol=0.1))
            print(f"  Symbol errors: {symbol_errors}/{len(data_symbols)}")
            print(f"  Symbols correct: {'✓' if symbol_errors == 0 else '✗'}")
        else:
            print(f"  Symbol length mismatch: {len(rx_data)} vs {len(data_symbols)}")

def visualize_spectrum_shifts():
    """Visualize how spectrum changes with different approaches."""
    print("\n" + "="*60)
    print("SPECTRUM VISUALIZATION")
    print("="*60)
    
    # Create simple test signal
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    test_bits = np.zeros(num_data_bits)  # All zeros for simplicity
    data_symbols = qam_mod(test_bits)
    ofdm_symbol = insert_pilots(data_symbols)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Frequency domain
    axes[0,0].stem(np.arange(params.N), np.abs(ofdm_symbol), basefmt=' ')
    axes[0,0].set_title('Original OFDM Symbol (Natural Order)')
    axes[0,0].set_xlabel('Subcarrier Index')
    
    ofdm_after_ifftshift = np.fft.ifftshift(ofdm_symbol)
    axes[0,1].stem(np.arange(params.N), np.abs(ofdm_after_ifftshift), basefmt=' ')
    axes[0,1].set_title('After ifftshift (Current Approach)')
    axes[0,1].set_xlabel('Subcarrier Index')
    
    ofdm_centered = np.fft.fftshift(ofdm_symbol)
    centered_indices = np.arange(-params.N//2, params.N//2)
    axes[0,2].stem(centered_indices, np.abs(ofdm_centered), basefmt=' ')
    axes[0,2].set_title('Centered Spectrum (fftshift)')
    axes[0,2].set_xlabel('Subcarrier Index (centered)')
    
    # Row 2: After receiver FFT + fftshift
    # Approach 1: Current
    time_1 = np.fft.ifft(ofdm_after_ifftshift) * np.sqrt(params.N)
    rx_freq_1 = np.fft.fft(time_1) / np.sqrt(params.N)
    rx_freq_1_centered = np.fft.fftshift(rx_freq_1)
    axes[1,0].stem(centered_indices, np.abs(rx_freq_1_centered), basefmt=' ')
    axes[1,0].set_title('Receiver: Current Approach')
    axes[1,0].set_xlabel('Subcarrier Index (centered)')
    
    # Approach 2: Fixed
    time_2 = np.fft.ifft(ofdm_symbol) * np.sqrt(params.N)
    rx_freq_2 = np.fft.fft(time_2) / np.sqrt(params.N)
    rx_freq_2_centered = np.fft.fftshift(rx_freq_2)
    axes[1,1].stem(centered_indices, np.abs(rx_freq_2_centered), basefmt=' ')
    axes[1,1].set_title('Receiver: Fixed Approach')
    axes[1,1].set_xlabel('Subcarrier Index (centered)')
    
    # Mark expected pilot positions
    pilot_carriers = np.array([-21, -7, 7, 21])
    for ax in axes[1,:]:
        for pc in pilot_carriers:
            ax.axvline(x=pc, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('transmitter_debug.png', dpi=150, bbox_inches='tight')
    print("Spectrum visualization saved to 'transmitter_debug.png'")

def main():
    """Run transmitter diagnostic."""
    print("TRANSMITTER DIAGNOSTIC")
    print("="*80)
    
    # Analyze pilot insertion
    ofdm_symbol, pilot_indices, data_indices = analyze_pilot_insertion()
    
    # Analyze IFFT shift issue
    analyze_ifft_shift_issue(ofdm_symbol, pilot_indices, data_indices)
    
    # Test both approaches
    test_both_approaches()
    
    # Visualize
    visualize_spectrum_shifts()
    
    print("\n" + "="*80)
    print("TRANSMITTER DIAGNOSTIC COMPLETE")
    print("="*80)
    
    print("\nKEY FINDINGS:")
    print("- The transmitter applies ifftshift to a natural-order OFDM symbol")
    print("- This scrambles the pilot/data positions before transmission")
    print("- The receiver expects pilots at specific centered positions")
    print("- Solution: Remove ifftshift from transmitter OR adjust pilot pattern")

if __name__ == "__main__":
    main()
