"""
Arithmetic coder for neural text compression.

Uses high-precision integer arithmetic (32-bit range) with proper
renormalization and underflow handling. The encoder and decoder are
perfectly symmetric — given the same sequence of CDFs, the decoder
recovers the exact symbol sequence the encoder consumed.
"""


class ArithmeticEncoder:
    """Encodes symbols into a compressed bitstream using arithmetic coding.

    Bits are packed into a bytearray on the fly instead of stored as
    individual Python ints, cutting memory from O(n_bits * 28 bytes) to
    O(n_bits / 8).
    """

    PRECISION = 32
    FULL = 1 << PRECISION          # 2^32
    HALF = 1 << (PRECISION - 1)    # 2^31
    QUARTER = 1 << (PRECISION - 2) # 2^30
    MAX_RANGE = FULL - 1           # 0xFFFFFFFF

    def __init__(self):
        self.low = 0
        self.high = self.MAX_RANGE
        self.pending_bits = 0
        self._buf = bytearray()
        self._cur_byte = 0
        self._bits_in_cur = 0
        self._total_bits = 0

    def _write_bit(self, bit: int):
        """Pack a single bit into the output bytearray."""
        self._cur_byte = (self._cur_byte << 1) | bit
        self._bits_in_cur += 1
        self._total_bits += 1
        if self._bits_in_cur == 8:
            self._buf.append(self._cur_byte)
            self._cur_byte = 0
            self._bits_in_cur = 0

    def _output_bit(self, bit: int):
        self._write_bit(bit)
        # Flush pending bits (opposite of the bit just emitted)
        for _ in range(self.pending_bits):
            self._write_bit(1 - bit)
        self.pending_bits = 0

    def encode_symbol(self, cdf, symbol_index: int):
        """Encode a single symbol given its CDF.

        Args:
            cdf: Cumulative distribution function. Supports both list[int]
                 and torch.Tensor (indexed with []). Length = num_symbols + 1.
                 cdf[0] = 0, cdf[-1] = total.
            symbol_index: Index of the symbol to encode (0-based).
        """
        total = int(cdf[-1])
        rng = self.high - self.low + 1

        sym_lo = int(cdf[symbol_index])
        sym_hi = int(cdf[symbol_index + 1])

        # Narrow the interval
        self.high = self.low + (rng * sym_hi) // total - 1
        self.low = self.low + (rng * sym_lo) // total

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

        # Pad to byte boundary
        while self._bits_in_cur != 0:
            self._write_bit(0)

        return bytes(self._buf)

    def get_bit_count(self) -> int:
        """Return number of bits written so far (approximate)."""
        return self._total_bits + self.pending_bits


class ArithmeticDecoder:
    """Decodes symbols from a compressed bitstream using arithmetic coding.

    Reads bits lazily from the compressed bytes instead of expanding
    every byte into 8 Python ints upfront.
    """

    PRECISION = 32
    FULL = 1 << PRECISION
    HALF = 1 << (PRECISION - 1)
    QUARTER = 1 << (PRECISION - 2)
    MAX_RANGE = FULL - 1

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_buf = 0
        self._bits_left = 0
        self.low = 0
        self.high = self.MAX_RANGE

        # Read initial value
        self.value = 0
        for _ in range(self.PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self._bits_left == 0:
            if self._byte_pos < len(self._data):
                self._bit_buf = self._data[self._byte_pos]
                self._byte_pos += 1
                self._bits_left = 8
            else:
                return 0  # Implicit trailing zeros
        self._bits_left -= 1
        return (self._bit_buf >> self._bits_left) & 1

    def decode_symbol(self, cdf) -> int:
        """Decode a single symbol given its CDF.

        Args:
            cdf: Same CDF format as encoder. Supports both list[int] and
                 torch.Tensor. Length = num_symbols + 1, cdf[0] = 0,
                 cdf[-1] = total.

        Returns:
            The symbol index (0-based).
        """
        total = int(cdf[-1])
        rng = self.high - self.low + 1

        # Find which symbol the current value falls into
        scaled_value = ((self.value - self.low + 1) * total - 1) // rng

        # Binary search for the symbol
        num_symbols = len(cdf) - 1
        lo, hi = 0, num_symbols - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if int(cdf[mid + 1]) <= scaled_value:
                lo = mid + 1
            else:
                hi = mid - 1
        symbol = lo

        sym_lo = int(cdf[symbol])
        sym_hi = int(cdf[symbol + 1])

        # Update range (must match encoder exactly)
        self.high = self.low + (rng * sym_hi) // total - 1
        self.low = self.low + (rng * sym_lo) // total

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
