"""
Model wrapper for SmolLM2-135M language model.

Handles model loading, device selection (CUDA with CPU fallback),
and probability distribution generation for arithmetic coding.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:
    """Wraps SmolLM2-135M for next-token probability prediction.

    Uses KV-cache for efficient incremental inference — each call
    after the first only processes the single new token, reusing
    cached key/value states from prior tokens. This keeps GPU memory
    constant regardless of sequence length.
    """

    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    MAX_CONTEXT = 2048
    # When context exceeds MAX_CONTEXT, drop this many old tokens at once.
    # This means we rebuild the cache once every SLIDE_CHUNK tokens instead
    # of every single token after hitting the limit.
    SLIDE_CHUNK = 512

    def __init__(self, model_name: str = None, verbose: bool = True):
        self.model_name = model_name or self.MODEL_NAME
        self.verbose = verbose
        self.device = self._select_device()
        if self.verbose:
            print(f"Using device: {self.device}", file=sys.stderr)
        self._load_model()
        self._past_key_values = None
        self._cache_len = 0

    def _select_device(self) -> str:
        """Select CUDA if available and working, otherwise CPU."""
        if torch.cuda.is_available():
            try:
                t = torch.tensor([1.0], device="cuda")
                del t
                return "cuda"
            except Exception:
                pass
        return "cpu"

    def _load_model(self):
        """Load model and tokenizer."""
        if self.verbose:
            print(f"Loading model {self.model_name}...", file=sys.stderr)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

        self.vocab_size = self.model.config.vocab_size
        if self.verbose:
            print(
                f"Model loaded: vocab_size={self.vocab_size}, "
                f"device={self.device}",
                file=sys.stderr,
            )

    def reset_cache(self):
        """Clear the KV-cache. Call when starting a new sequence."""
        self._past_key_values = None
        self._cache_len = 0
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def get_probs(self, token_ids: list[int]) -> torch.Tensor:
        """Get next-token probability distribution given context.

        Uses KV-cache for incremental inference. On the first call (or
        after reset_cache), processes the full context. On subsequent
        calls where only one token was appended, processes just that
        single new token using the cached states.

        When the context exceeds MAX_CONTEXT, it drops SLIDE_CHUNK old
        tokens at once and rebuilds the cache from the remaining context.
        This amortizes the rebuild cost: one full forward pass per
        SLIDE_CHUNK tokens instead of per token.

        Args:
            token_ids: List of token IDs as context. Can be empty,
                       in which case a BOS/default context is used.

        Returns:
            Tensor of shape (vocab_size,) with probabilities summing to 1.
        """
        if len(token_ids) == 0:
            if self.tokenizer.bos_token_id is not None:
                token_ids = [self.tokenizer.bos_token_id]
            else:
                token_ids = [0]

        # Chunked sliding window: when we exceed MAX_CONTEXT, drop
        # SLIDE_CHUNK tokens at once so we can use incremental decoding
        # for the next SLIDE_CHUNK calls before needing to rebuild again.
        if len(token_ids) > self.MAX_CONTEXT:
            keep = self.MAX_CONTEXT - self.SLIDE_CHUNK
            token_ids = token_ids[-keep:]
            self._past_key_values = None
            self._cache_len = 0

        ctx_len = len(token_ids)

        if (self._past_key_values is not None
                and ctx_len == self._cache_len + 1):
            # Incremental: only feed the last token
            new_ids = torch.tensor(
                [[token_ids[-1]]], dtype=torch.long, device=self.device
            )
            outputs = self.model(
                new_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
        else:
            # Full context (first call or cache miss)
            self._past_key_values = None
            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self.device
            )
            outputs = self.model(
                input_ids,
                use_cache=True,
            )

        self._past_key_values = outputs.past_key_values
        self._cache_len = ctx_len

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=0)
        return probs.cpu()
