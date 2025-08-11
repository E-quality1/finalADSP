"""
QAM Modulation/Demodulation Diagnostic

This script isolates and debugs the QAM modulation/demodulation issues
identified by the test bench.
"""
import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import QAMModem
from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod

def test_qam_basic():
    """Test basic QAM modulation/demodulation."""
    print("="*60)
    print("QAM BASIC TEST")
    print("="*60)
    
    # Test with small number of bits
    test_bits = np.array([0, 1, 1, 0, 1, 1])  # 6 bits = 1 symbol for 64-QAM
    print(f"Input bits: {test_bits}")
    print(f"Number of input bits: {len(test_bits)}")
    print(f"Bits per symbol: {params.bits_per_symbol}")
    
    # Modulate
    symbols = qam_mod(test_bits)
    print(f"Modulated symbols: {symbols}")
    print(f"Number of symbols: {len(symbols)}")
    
    # Demodulate
    demod_bits = qam_demod(symbols)
    print(f"Demodulated bits: {demod_bits}")
    print(f"Number of output bits: {len(demod_bits)}")
    
    # Check if they match
    if len(demod_bits) == len(test_bits):
        errors = np.sum(demod_bits != test_bits)
        print(f"Bit errors: {errors}")
        print(f"Match: {'✓ YES' if errors == 0 else '✗ NO'}")
    else:
        print(f"Length mismatch: input {len(test_bits)}, output {len(demod_bits)}")
    
    return symbols, demod_bits

def test_qam_lengths():
    """Test QAM with different bit lengths."""
    print("\n" + "="*60)
    print("QAM LENGTH TEST")
    print("="*60)
    
    # Test different bit lengths
    test_lengths = [6, 12, 18, 24, 30, 36]  # Multiples of 6 for 64-QAM
    
    for num_bits in test_lengths:
        test_bits = np.random.randint(0, 2, num_bits)
        
        try:
            symbols = qam_mod(test_bits)
            demod_bits = qam_demod(symbols)
            
            expected_symbols = num_bits // params.bits_per_symbol
            expected_bits = len(symbols) * params.bits_per_symbol
            
            print(f"Bits: {num_bits:2d} → Symbols: {len(symbols):2d} (exp: {expected_symbols:2d}) → Bits: {len(demod_bits):2d} (exp: {expected_bits:2d})")
            
            if len(demod_bits) == num_bits:
                errors = np.sum(demod_bits != test_bits)
                print(f"  Errors: {errors}/{num_bits} ({'✓' if errors == 0 else '✗'})")
            else:
                print(f"  Length mismatch! ✗")
                
        except Exception as e:
            print(f"  Error: {e}")

def test_commpy_directly():
    """Test commpy QAMModem directly."""
    print("\n" + "="*60)
    print("COMMPY DIRECT TEST")
    print("="*60)
    
    # Create modem directly
    modem = QAMModem(params.mod_order)
    print(f"QAM order: {params.mod_order}")
    print(f"Bits per symbol: {modem.num_bits_symbol}")
    
    # Test with exact number of bits
    num_bits = 12  # 2 symbols for 64-QAM
    test_bits = np.random.randint(0, 2, num_bits)
    
    print(f"Test bits ({len(test_bits)}): {test_bits}")
    
    # Modulate
    symbols = modem.modulate(test_bits)
    print(f"Symbols ({len(symbols)}): {symbols}")
    
    # Demodulate
    demod_bits = modem.demodulate(symbols, 'hard')
    print(f"Demod bits ({len(demod_bits)}): {demod_bits}")
    
    # Check
    if len(demod_bits) == len(test_bits):
        errors = np.sum(demod_bits != test_bits)
        print(f"Errors: {errors}/{len(test_bits)}")
        print(f"Status: {'✓ PASS' if errors == 0 else '✗ FAIL'}")
    else:
        print(f"Length mismatch: {len(test_bits)} → {len(demod_bits)}")
        print("Status: ✗ FAIL")

def test_constellation_plot():
    """Plot QAM constellation."""
    print("\n" + "="*60)
    print("QAM CONSTELLATION PLOT")
    print("="*60)
    
    # Generate all possible symbols
    modem = QAMModem(params.mod_order)
    num_symbols = 2**params.bits_per_symbol
    
    # Generate all possible bit combinations
    all_symbols = []
    for i in range(num_symbols):
        bits = np.array([(i >> j) & 1 for j in range(params.bits_per_symbol-1, -1, -1)])
        symbol = modem.modulate(bits)[0]
        all_symbols.append(symbol)
    
    all_symbols = np.array(all_symbols)
    
    # Plot constellation
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(all_symbols), np.imag(all_symbols), s=100, alpha=0.7)
    
    # Add labels
    for i, symbol in enumerate(all_symbols):
        bits = [(i >> j) & 1 for j in range(params.bits_per_symbol-1, -1, -1)]
        bit_str = ''.join(map(str, bits))
        plt.annotate(bit_str, (np.real(symbol), np.imag(symbol)), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.title(f'{params.mod_order}-QAM Constellation')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig('qam_constellation.png', dpi=150, bbox_inches='tight')
    print("Constellation saved to 'qam_constellation.png'")

def test_noisy_demodulation():
    """Test demodulation with noise."""
    print("\n" + "="*60)
    print("NOISY DEMODULATION TEST")
    print("="*60)
    
    # Generate test data
    num_bits = 60  # 10 symbols for 64-QAM
    test_bits = np.random.randint(0, 2, num_bits)
    
    # Modulate
    clean_symbols = qam_mod(test_bits)
    
    # Add different levels of noise
    snr_values = [30, 20, 15, 10, 5]
    
    for snr_db in snr_values:
        # Add AWGN
        noise_var = 10**(-snr_db/10)
        noise = np.sqrt(noise_var/2) * (np.random.randn(len(clean_symbols)) + 1j*np.random.randn(len(clean_symbols)))
        noisy_symbols = clean_symbols + noise
        
        # Demodulate
        demod_bits = qam_demod(noisy_symbols)
        
        # Calculate BER
        if len(demod_bits) == len(test_bits):
            errors = np.sum(demod_bits != test_bits)
            ber = errors / len(test_bits)
            print(f"SNR {snr_db:2d} dB: BER = {ber:.6f} ({errors}/{len(test_bits)} errors)")
        else:
            print(f"SNR {snr_db:2d} dB: Length mismatch")

def main():
    """Run all QAM diagnostic tests."""
    print("QAM MODULATION/DEMODULATION DIAGNOSTIC")
    print("="*80)
    
    test_qam_basic()
    test_qam_lengths()
    test_commpy_directly()
    test_constellation_plot()
    test_noisy_demodulation()
    
    print("\n" + "="*80)
    print("QAM DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
