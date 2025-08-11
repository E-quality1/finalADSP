"""
Detailed CFO Estimation Debugging
This module provides detailed analysis of the CFO estimation process
to identify the exact source of errors.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from ofdm.system import params
from ofdm.transmitter import qam_mod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def create_perfect_test_symbol():
    """Create a perfect test symbol with known pilot values and no CFO."""
    # Create OFDM symbol in frequency domain with known values
    ofdm_symbol_freq = np.zeros(params.N, dtype=complex)
    
    # Insert pilots at correct locations
    pilot_indices = np.where(params.pilot_pattern)[0]
    data_indices = np.where(~params.pilot_pattern)[0]
    
    # Use known pilot values
    ofdm_symbol_freq[pilot_indices] = params.pilot_value
    # Use random data values for realistic testing
    np.random.seed(42)  # For reproducible results
    num_data_symbols = len(data_indices)
    data_bits = np.random.randint(0, 2, num_data_symbols * params.bits_per_symbol)
    data_symbols = qam_mod(data_bits)
    ofdm_symbol_freq[data_indices] = data_symbols
    
    # Convert to centered spectrum
    ofdm_symbol_freq_centered = np.fft.fftshift(ofdm_symbol_freq)
    
    return ofdm_symbol_freq_centered

def analyze_pilot_extraction():
    """Analyze how pilots are extracted from centered spectrum."""
    print("=" * 60)
    print("ANALYZING PILOT EXTRACTION")
    print("=" * 60)
    
    # Create perfect test symbol
    test_symbol = create_perfect_test_symbol()
    
    # Method 1: Using natural order indices (current approach in estimator)
    pilot_carriers = np.array([-21, -7, 7, 21])
    pilot_indices_centered = (pilot_carriers + params.N // 2) % params.N
    
    print(f"Pilot carriers (relative to center): {pilot_carriers}")
    print(f"Pilot indices in centered spectrum: {pilot_indices_centered}")
    
    # Extract pilots using these indices
    extracted_pilots = test_symbol[pilot_indices_centered]
    print(f"Extracted pilot values: {extracted_pilots}")
    print(f"Expected pilot values: {[params.pilot_value] * len(pilot_carriers)}")
    
    # Check if extraction is correct
    correct = np.allclose(extracted_pilots, params.pilot_value)
    print(f"Pilot extraction correct: {correct}")
    
    if not correct:
        print("ERROR: Pilot extraction is incorrect!")
        # Debug by showing the entire spectrum
        print(f"Full spectrum around pilots:")
        for i, idx in enumerate(pilot_indices_centered):
            print(f"  Pilot {i} at index {idx}: {test_symbol[idx]}")

def test_correlation_manually():
    """Manually test the correlation logic step by step."""
    print("\n" + "=" * 60)
    print("MANUAL CORRELATION TEST")
    print("=" * 60)
    
    # Create test symbol with known CFO
    true_cfo = 1
    perfect_symbol = create_perfect_test_symbol()
    test_symbol = np.roll(perfect_symbol, true_cfo)  # Apply CFO
    
    print(f"Testing with true CFO: {true_cfo}")
    
    # Create ideal pilot template
    pilot_carriers = np.array([-21, -7, 7, 21])
    pilot_indices_centered = (pilot_carriers + params.N // 2) % params.N
    
    ideal_template = np.zeros(params.N, dtype=complex)
    ideal_template[pilot_indices_centered] = params.pilot_value
    
    print(f"Ideal template has {np.sum(np.abs(ideal_template) > 0)} non-zero elements")
    
    # Test different shifts
    max_cfo = 3
    print(f"\nTesting shifts from {-max_cfo} to {max_cfo}:")
    
    for shift in range(-max_cfo, max_cfo + 1):
        shifted_rx = np.roll(test_symbol, shift)
        
        # Method 1: Current approach
        corr1 = np.abs(np.vdot(ideal_template, shifted_rx))
        
        # Method 2: Alternative approach
        corr2 = np.abs(np.vdot(shifted_rx, ideal_template))
        
        # Method 3: Manual dot product
        corr3 = np.abs(np.sum(np.conj(ideal_template) * shifted_rx))
        
        print(f"  Shift {shift:2d}: corr1={corr1:.3f}, corr2={corr2:.3f}, corr3={corr3:.3f}")
        
        # Check if this shift corrects the CFO
        if shift == -true_cfo:
            print(f"    ^ This should be the peak (shift = -true_cfo = {-true_cfo})")

def visualize_spectrum_shift():
    """Visualize how spectrum shifting affects pilot positions."""
    print("\n" + "=" * 60)
    print("VISUALIZING SPECTRUM SHIFT")
    print("=" * 60)
    
    # Create perfect test symbol
    perfect_symbol = create_perfect_test_symbol()
    
    # Apply different CFO values
    cfos_to_test = [-2, -1, 0, 1, 2]
    
    fig, axes = plt.subplots(len(cfos_to_test), 1, figsize=(12, 10))
    
    for i, cfo in enumerate(cfos_to_test):
        shifted_symbol = np.roll(perfect_symbol, cfo)
        
        # Plot magnitude spectrum
        freqs = np.arange(-params.N//2, params.N//2)
        axes[i].stem(freqs, np.abs(shifted_symbol), basefmt=' ')
        axes[i].set_title(f'CFO = {cfo}')
        axes[i].set_ylabel('Magnitude')
        axes[i].grid(True, alpha=0.3)
        
        # Mark expected pilot positions
        pilot_carriers = np.array([-21, -7, 7, 21])
        for pc in pilot_carriers:
            axes[i].axvline(x=pc, color='r', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Subcarrier Index (centered)')
    plt.tight_layout()
    plt.savefig('spectrum_shift_analysis.png', dpi=150, bbox_inches='tight')
    print("Spectrum shift analysis saved as 'spectrum_shift_analysis.png'")

if __name__ == "__main__":
    analyze_pilot_extraction()
    test_correlation_manually()
    visualize_spectrum_shift()
    
    print("\n" + "=" * 60)
    print("DETAILED CFO DEBUGGING COMPLETE")
    print("=" * 60)
