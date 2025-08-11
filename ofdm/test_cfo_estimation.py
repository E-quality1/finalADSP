"""
Isolated CFO Estimation Testing
This module tests the integer and fractional CFO estimators in isolation
to identify the root cause of estimation errors.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from ofdm.system import params
from ofdm.transmitter import qam_mod
from ofdm.equalization import estimate_integer_cfo_pilots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def create_test_symbol_with_integer_cfo(integer_cfo):
    """Create a test OFDM symbol with known integer CFO for testing."""
    # Generate random data bits
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    data_bits = np.random.randint(0, 2, num_data_bits)
    
    # Modulate to 64-QAM
    data_symbols = qam_mod(data_bits)
    
    # Create OFDM symbol in frequency domain
    ofdm_symbol_freq = np.zeros(params.N, dtype=complex)
    
    # Insert pilots at correct locations
    pilot_indices = np.where(params.pilot_pattern)[0]
    data_indices = np.where(~params.pilot_pattern)[0]
    
    ofdm_symbol_freq[pilot_indices] = params.pilot_value
    ofdm_symbol_freq[data_indices] = data_symbols
    
    # Convert to centered spectrum first
    ofdm_symbol_freq_centered = np.fft.fftshift(ofdm_symbol_freq)
    
    # Apply integer CFO by shifting in the centered domain
    ofdm_symbol_freq_centered_shifted = np.roll(ofdm_symbol_freq_centered, integer_cfo)
    
    return ofdm_symbol_freq_centered_shifted

def test_integer_cfo_estimator():
    """Test the integer CFO estimator with known integer offsets."""
    print("=" * 60)
    print("TESTING INTEGER CFO ESTIMATOR")
    print("=" * 60)
    
    test_cfos = [-3, -2, -1, 0, 1, 2, 3]
    
    for true_cfo in test_cfos:
        # Create test symbol with known integer CFO
        rx_symbol_centered = create_test_symbol_with_integer_cfo(true_cfo)
        
        # Estimate integer CFO
        estimated_cfo = estimate_integer_cfo_pilots(rx_symbol_centered)
        
        # Check accuracy
        error = estimated_cfo - true_cfo
        status = "✓ PASS" if error == 0 else "✗ FAIL"
        
        print(f"True CFO: {true_cfo:2d}, Estimated: {estimated_cfo:2d}, Error: {error:2d} {status}")

def analyze_pilot_pattern():
    """Analyze the pilot pattern to verify our understanding."""
    print("\n" + "=" * 60)
    print("ANALYZING PILOT PATTERN")
    print("=" * 60)
    
    # Get pilot indices in natural order
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    print(f"Pilot indices (natural order): {pilot_indices_natural}")
    
    # Convert to centered indices
    pilot_carriers = np.array([-21, -7, 7, 21])  # Expected centered positions
    pilot_indices_centered_expected = (pilot_carriers + params.N // 2) % params.N
    print(f"Expected centered indices: {pilot_indices_centered_expected}")
    
    # Check if they match
    match = np.array_equal(np.sort(pilot_indices_natural), np.sort(pilot_indices_centered_expected))
    print(f"Pilot patterns match: {match}")
    
    if not match:
        print("WARNING: Pilot pattern mismatch detected!")
        print(f"Natural order indices: {np.sort(pilot_indices_natural)}")
        print(f"Expected centered indices: {np.sort(pilot_indices_centered_expected)}")

def visualize_correlation_function():
    """Visualize the correlation function for a known integer CFO."""
    print("\n" + "=" * 60)
    print("VISUALIZING CORRELATION FUNCTION")
    print("=" * 60)
    
    true_cfo = 2  # Test with CFO of +2
    rx_symbol_centered = create_test_symbol_with_integer_cfo(true_cfo)
    
    # Recreate the correlation logic from estimate_integer_cfo_pilots
    pilot_carriers = np.array([-21, -7, 7, 21])
    pilot_indices_centered = (pilot_carriers + params.N // 2) % params.N
    
    ideal_pilot_template = np.zeros(params.N, dtype=complex)
    ideal_pilot_template[pilot_indices_centered] = params.pilot_value
    
    max_cfo = 8
    shifts = np.arange(-max_cfo, max_cfo + 1)
    correlations = []
    
    for shift in shifts:
        shifted_rx_freq = np.roll(rx_symbol_centered, shift)
        corr = np.abs(np.vdot(ideal_pilot_template, shifted_rx_freq))
        correlations.append(corr)
    
    correlations = np.array(correlations)
    best_shift_idx = np.argmax(correlations)
    best_shift = shifts[best_shift_idx]
    estimated_cfo = -best_shift
    
    print(f"True CFO: {true_cfo}")
    print(f"Best shift: {best_shift}")
    print(f"Estimated CFO: {estimated_cfo}")
    print(f"Estimation error: {estimated_cfo - true_cfo}")
    
    # Plot correlation function
    plt.figure(figsize=(10, 6))
    plt.plot(shifts, correlations, 'b-o', linewidth=2, markersize=8)
    plt.axvline(x=best_shift, color='r', linestyle='--', label=f'Best shift: {best_shift}')
    plt.axvline(x=-true_cfo, color='g', linestyle='--', label=f'Expected shift: {-true_cfo}')
    plt.xlabel('Shift (subcarriers)')
    plt.ylabel('Correlation Magnitude')
    plt.title(f'Integer CFO Correlation Function (True CFO = {true_cfo})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cfo_correlation_function.png', dpi=150, bbox_inches='tight')
    print("Correlation function plot saved as 'cfo_correlation_function.png'")

if __name__ == "__main__":
    # Run all tests
    analyze_pilot_pattern()
    test_integer_cfo_estimator()
    visualize_correlation_function()
    
    print("\n" + "=" * 60)
    print("CFO ESTIMATION TESTING COMPLETE")
    print("=" * 60)
