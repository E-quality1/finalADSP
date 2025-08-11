"""
Final CFO Estimation Debugging
This module creates a test that exactly matches the transmitter/receiver chain
to identify and fix the CFO estimation issues.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from ofdm.system import params
from ofdm.transmitter import qam_mod, insert_pilots, ifft_with_cp
from ofdm.equalization import estimate_integer_cfo_pilots, remove_cp_and_fft

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def create_realistic_test_signal(integer_cfo=0):
    """Create a test signal using the actual transmitter chain."""
    # Generate random data bits
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    data_bits = np.random.randint(0, 2, num_data_bits)
    
    # Modulate to 64-QAM
    data_symbols = qam_mod(data_bits)
    
    # Insert pilots (this uses the actual transmitter logic)
    ofdm_symbol_with_pilots = insert_pilots(data_symbols)
    
    # Apply IFFT and CP
    time_domain_signal = ifft_with_cp(ofdm_symbol_with_pilots)
    
    # Simulate integer CFO in time domain (frequency shift)
    if integer_cfo != 0:
        # Apply frequency shift in time domain
        n = np.arange(len(time_domain_signal))
        cfo_phase = 2 * np.pi * integer_cfo * n / params.N
        time_domain_signal = time_domain_signal * np.exp(1j * cfo_phase)
    
    return time_domain_signal, data_symbols

def test_realistic_cfo_estimation():
    """Test CFO estimation using the actual transmitter/receiver chain."""
    print("=" * 60)
    print("REALISTIC CFO ESTIMATION TEST")
    print("=" * 60)
    
    test_cfos = [-2, -1, 0, 1, 2]
    
    for true_cfo in test_cfos:
        # Create realistic test signal
        rx_signal, original_data = create_realistic_test_signal(integer_cfo=true_cfo)
        
        # Use the actual receiver chain
        rx_symbol_freq_centered = remove_cp_and_fft(rx_signal, 0)  # No timing offset
        
        # Estimate integer CFO
        estimated_cfo = estimate_integer_cfo_pilots(rx_symbol_freq_centered)
        
        # Check accuracy
        error = estimated_cfo - true_cfo
        status = "✓ PASS" if abs(error) <= 1 else "✗ FAIL"  # Allow ±1 error for now
        
        print(f"True CFO: {true_cfo:2d}, Estimated: {estimated_cfo:2d}, Error: {error:2d} {status}")

def analyze_transmitter_pilot_placement():
    """Analyze how the transmitter places pilots."""
    print("\n" + "=" * 60)
    print("ANALYZING TRANSMITTER PILOT PLACEMENT")
    print("=" * 60)
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    data_bits = np.random.randint(0, 2, num_data_bits)
    data_symbols = qam_mod(data_bits)
    
    # Insert pilots using transmitter logic
    ofdm_symbol_with_pilots = insert_pilots(data_symbols)
    
    print(f"OFDM symbol length: {len(ofdm_symbol_with_pilots)}")
    print(f"Number of pilots: {np.sum(params.pilot_pattern)}")
    print(f"Number of data symbols: {len(data_symbols)}")
    
    # Find where pilots are located
    pilot_indices = np.where(params.pilot_pattern)[0]
    print(f"Pilot indices (natural order): {pilot_indices}")
    
    # Check pilot values in the symbol
    pilot_values = ofdm_symbol_with_pilots[pilot_indices]
    print(f"Pilot values in symbol: {pilot_values}")
    print(f"Expected pilot value: {params.pilot_value}")
    
    # Check if pilots are correctly placed
    pilots_correct = np.allclose(pilot_values, params.pilot_value)
    print(f"Pilots correctly placed: {pilots_correct}")
    
    # Apply fftshift and check again
    ofdm_symbol_centered = np.fft.fftshift(ofdm_symbol_with_pilots)
    pilot_values_centered = ofdm_symbol_centered[pilot_indices]
    print(f"Pilot values after fftshift: {pilot_values_centered}")

def debug_correlation_step_by_step():
    """Debug the correlation process step by step."""
    print("\n" + "=" * 60)
    print("DEBUGGING CORRELATION STEP BY STEP")
    print("=" * 60)
    
    # Create test signal with known CFO
    true_cfo = 1
    rx_signal, _ = create_realistic_test_signal(integer_cfo=true_cfo)
    rx_symbol_freq_centered = remove_cp_and_fft(rx_signal, 0)
    
    print(f"Testing with true CFO: {true_cfo}")
    
    # Create ideal pilot template (using the same logic as the estimator)
    pilot_indices = np.where(params.pilot_pattern)[0]
    ideal_template = np.zeros(params.N, dtype=complex)
    ideal_template[pilot_indices] = params.pilot_value
    
    print(f"Pilot indices: {pilot_indices}")
    print(f"Ideal template non-zero at: {np.where(np.abs(ideal_template) > 0)[0]}")
    
    # Extract pilots from received signal
    rx_pilots = rx_symbol_freq_centered[pilot_indices]
    print(f"Received pilot values: {rx_pilots}")
    
    # Test correlation for different shifts
    max_cfo = 3
    correlations = []
    
    for shift in range(-max_cfo, max_cfo + 1):
        shifted_rx = np.roll(rx_symbol_freq_centered, shift)
        corr = np.abs(np.vdot(ideal_template, shifted_rx))
        correlations.append(corr)
        
        # Extract pilots from shifted signal
        shifted_pilots = shifted_rx[pilot_indices]
        pilot_match = np.abs(np.vdot(params.pilot_value * np.ones(len(pilot_indices)), shifted_pilots))
        
        print(f"  Shift {shift:2d}: corr={corr:.3f}, pilot_match={pilot_match:.3f}")
        
        if shift == -true_cfo:
            print(f"    ^ Expected peak at shift = -true_cfo = {-true_cfo}")
    
    # Find the best shift
    best_shift_idx = np.argmax(correlations)
    best_shift = best_shift_idx - max_cfo
    estimated_cfo = -best_shift
    
    print(f"\nBest shift: {best_shift}")
    print(f"Estimated CFO: {estimated_cfo}")
    print(f"Error: {estimated_cfo - true_cfo}")

if __name__ == "__main__":
    analyze_transmitter_pilot_placement()
    test_realistic_cfo_estimation()
    debug_correlation_step_by_step()
    
    print("\n" + "=" * 60)
    print("FINAL CFO DEBUGGING COMPLETE")
    print("=" * 60)
