"""Validated Llama dimensions and parameter counts; no ML dependencies required."""
from dataclasses import asdict, dataclass


def llama_parameter_count(config):
    """Count dense Llama parameters, including optional biases and tied embeddings."""
    h = config['hidden_size']
    intermediate = config['intermediate_size']
    heads = config['num_attention_heads']
    kv_heads = config.get('num_key_value_heads') or heads
    head_dim = config.get('head_dim') or h // heads
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    attention = 2 * h * q_width + 2 * h * kv_width
    mlp = 3 * h * intermediate
    if config.get('attention_bias', False):
        attention += q_width + 2 * kv_width + h
    if config.get('mlp_bias', False):
        mlp += 2 * intermediate + h
    embeddings = config['vocab_size'] * h * (1 if config.get('tie_word_embeddings', False) else 2)
    return config['num_hidden_layers'] * (attention + mlp + 2 * h) + h + embeddings


@dataclass(frozen=True)
class Architecture:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 4096

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.hidden_size % self.num_attention_heads:
            raise ValueError('hidden_size must be divisible by num_attention_heads')
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads')
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError('The attention head dimension must be even for rotary embeddings')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError('Architecture JSON must be an object')
        required = set(cls.__dataclass_fields__) - {'max_position_embeddings'}
        missing = required - data.keys()
        unknown = data.keys() - cls.__dataclass_fields__.keys()
        if missing or unknown:
            raise ValueError(f'Invalid architecture fields: missing={sorted(missing)}, unknown={sorted(unknown)}')
        return cls(**data)

    def parameter_count(self, vocab_size):
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError('vocab_size must be a positive integer')
        return llama_parameter_count({**self.to_dict(), 'vocab_size': vocab_size})

    def to_dict(self):
        return asdict(self)


# Approximate names at a vocabulary of 32,768. These are local experimental
# configurations, not the architectures or weights of pretrained vendor models.
PRESETS = {
    '20m': Architecture(256, 768, 4, 8, 4),
    '125m': Architecture(768, 2048, 11, 12, 12),
    '350m': Architecture(1024, 2816, 24, 16, 8),
    '1b': Architecture(2048, 5504, 20, 16, 4),
    '3b': Architecture(2560, 6912, 41, 20, 4),
    '4b': Architecture(3072, 8192, 38, 24, 8),
    '7b': Architecture(4096, 11008, 38, 32, 8),
    '13b': Architecture(5120, 13824, 46, 40, 8),
}
