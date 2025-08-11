"""
Helper utility functions
"""
import numpy as np

def ber_count(tx_bits, rx_bits):
    """Calculate bit error rate"""
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)
    return ber, errors

def mse(x, y):
    """Mean squared error"""
    return np.mean(np.abs(x - y) ** 2)
