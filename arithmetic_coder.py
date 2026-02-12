"""
Arithmetic coder for neural text compression.

Uses high-precision integer arithmetic (32-bit range) with proper
renormalization and underflow handling. The encoder and decoder are
perfectly symmetric — given the same sequence of CDFs, the decoder
recovers the exact symbol sequence the encoder consumed.
"""


class ArithmeticEncoder:
    """Encodes symbols into a compressed bitstream using arithmetic coding."""

    PRECISION = 32
    FULL = 1 << PRECISION          # 2^32
    HALF = 1 << (PRECISION - 1)    # 2^31
    QUARTER = 1 << (PRECISION - 2) # 2^30
    MAX_RANGE = FULL - 1           # 0xFFFFFFFF

    def __init__(self):
        self.low = 0
        self.high = self.MAX_RANGE
        self.pending_bits = 0
        self.bits = []

    def _output_bit(self, bit: int):
        self.bits.append(bit)
        # Flush pending bits (opposite of the bit just emitted)
        while self.pending_bits > 0:
            self.bits.append(1 - bit)
            self.pending_bits -= 1

    def encode_symbol(self, cdf: list[int], symbol_index: int):
        """Encode a single symbol given its CDF.

        Args:
            cdf: Cumulative distribution function as a list of integers.
                 Length = num_symbols + 1. cdf[0] = 0, cdf[-1] = total.
                 P(symbol i) is proportional to cdf[i+1] - cdf[i].
            symbol_index: Index of the symbol to encode (0-based).
        """
        total = cdf[-1]
        rng = self.high - self.low + 1

        # Narrow the interval
        self.high = self.low + (rng * cdf[symbol_index + 1]) // total - 1
        self.low = self.low + (rng * cdf[symbol_index]) // total

        # Renormalize
        while True:
            if self.high < self.HALF:
                # Both in lower half — output 0
                self._output_bit(0)
                self.low = self.low << 1
                self.high = (self.high << 1) | 1
            elif self.low >= self.HALF:
                # Both in upper half — output 1
                self._output_bit(1)
                self.low = (self.low - self.HALF) << 1
                self.high = ((self.high - self.HALF) << 1) | 1
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                # Underflow / near-convergence
                self.pending_bits += 1
                self.low = (self.low - self.QUARTER) << 1
                self.high = ((self.high - self.QUARTER) << 1) | 1
            else:
                break

        # Keep values in range
        self.low &= self.MAX_RANGE
        self.high &= self.MAX_RANGE

    def finish(self) -> bytes:
        """Finalize encoding and return compressed data as bytes."""
        # Flush remaining state
        self.pending_bits += 1
        if self.low < self.QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)

        # Convert bits to bytes
        # Pad to byte boundary
        while len(self.bits) % 8 != 0:
            self.bits.append(0)

        result = bytearray()
        for i in range(0, len(self.bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.bits[i + j]
            result.append(byte)
        return bytes(result)

    def get_bit_count(self) -> int:
        """Return number of bits written so far (approximate)."""
        return len(self.bits) + self.pending_bits


class ArithmeticDecoder:
    """Decodes symbols from a compressed bitstream using arithmetic coding."""

    PRECISION = 32
    FULL = 1 << PRECISION
    HALF = 1 << (PRECISION - 1)
    QUARTER = 1 << (PRECISION - 2)
    MAX_RANGE = FULL - 1

    def __init__(self, data: bytes):
        self.bits = []
        for byte in data:
            for i in range(7, -1, -1):
                self.bits.append((byte >> i) & 1)
        self.bit_pos = 0
        self.low = 0
        self.high = self.MAX_RANGE

        # Read initial value
        self.value = 0
        for _ in range(self.PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self.bit_pos < len(self.bits):
            bit = self.bits[self.bit_pos]
            self.bit_pos += 1
            return bit
        return 0  # Implicit trailing zeros

    def decode_symbol(self, cdf: list[int]) -> int:
        """Decode a single symbol given its CDF.

        Args:
            cdf: Same CDF format as encoder — list of integers,
                 length = num_symbols + 1, cdf[0] = 0, cdf[-1] = total.

        Returns:
            The symbol index (0-based).
        """
        total = cdf[-1]
        rng = self.high - self.low + 1

        # Find which symbol the current value falls into
        scaled_value = ((self.value - self.low + 1) * total - 1) // rng

        # Binary search for the symbol
        symbol = 0
        num_symbols = len(cdf) - 1
        lo, hi = 0, num_symbols - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if cdf[mid + 1] <= scaled_value:
                lo = mid + 1
            else:
                hi = mid - 1
        symbol = lo

        # Update range (must match encoder exactly)
        self.high = self.low + (rng * cdf[symbol + 1]) // total - 1
        self.low = self.low + (rng * cdf[symbol]) // total

        # Renormalize (must match encoder exactly)
        while True:
            if self.high < self.HALF:
                self.low = self.low << 1
                self.high = (self.high << 1) | 1
                self.value = (self.value << 1) | self._read_bit()
            elif self.low >= self.HALF:
                self.low = (self.low - self.HALF) << 1
                self.high = ((self.high - self.HALF) << 1) | 1
                self.value = ((self.value - self.HALF) << 1) | self._read_bit()
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.low = (self.low - self.QUARTER) << 1
                self.high = ((self.high - self.QUARTER) << 1) | 1
                self.value = ((self.value - self.QUARTER) << 1) | self._read_bit()
            else:
                break

        self.low &= self.MAX_RANGE
        self.high &= self.MAX_RANGE
        self.value &= self.MAX_RANGE

        return symbol
