__all__ = [
    'hamming_distance'
]


def hamming_distance(vector1: str, vector2: str) -> int:
    """
    Calculate the Hamming distance between two hexadecimal strings.

    The Hamming distance is the number of positions at which the corresponding bits are different.
    This function first converts the hexadecimal strings to integers and then uses the bitwise XOR operation
    to find the positions where the bits differ. Finally, it counts the number of set bits in the result.

    Args:
        vector1 (str): The first hexadecimal string representing a binary vector.
        vector2 (str): The second hexadecimal string representing a binary vector.

    Returns:
        int: The Hamming distance between the two vectors.
    """
    return (int(vector1, 16) ^ int(vector2, 16)).bit_count()
