"""Serializacao e desserializacao de modelos via pickle."""

import pickle


def serialize_model(model) -> bytes:
    """Serializa modelo para bytes (pickle)."""
    return pickle.dumps(model)


def deserialize_model(raw_bytes: bytes):
    """Desserializa modelo a partir de bytes (pickle)."""
    return pickle.loads(raw_bytes)
