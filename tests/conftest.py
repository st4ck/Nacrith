import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_wrapper import ModelWrapper
from compressor import NeuralCompressor


@pytest.fixture(scope="session")
def model_wrapper():
    return ModelWrapper(verbose=False)


@pytest.fixture(scope="session")
def compressor(model_wrapper):
    return NeuralCompressor(model=model_wrapper, verbose=False)


@pytest.fixture(scope="session")
def sample_texts():
    return {
        "short": "Hello, World!",
        "medium": (
            "The quick brown fox jumps over the lazy dog. "
            "This is a medium-length text sample that contains multiple sentences. "
            "It should be sufficient for testing basic compression and decompression."
        ),
        "long": (
            "In the beginning, there was nothing but darkness and silence. "
            "Then, slowly but surely, the universe began to take shape. "
            "Stars ignited across the cosmos, galaxies formed and collided, "
            "and planets coalesced from the debris of dying suns. On one such "
            "planet, in a small corner of a vast galaxy, life emerged. At first, "
            "it was simple — single-celled organisms floating in primordial seas. "
            "But over billions of years, these simple forms evolved into an "
            "incredible diversity of life. From the depths of the ocean to the "
            "highest mountains, from microscopic bacteria to massive whales, "
            "life found a way to adapt and thrive in every corner of the planet. "
            "And among all these forms of life, one species eventually developed "
            "the ability to contemplate its own existence, to wonder about the "
            "stars above, and to ask the eternal questions: Who are we? Where "
            "did we come from? And where are we going?"
        ),
    }
