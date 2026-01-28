"""Unit tests for customer behavior sequence encoder."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from ragrec.embeddings.sequence import BehaviorEncoder, CustomerBehaviorEncoder


def test_behavior_encoder_initialization():
    """Test BehaviorEncoder initialization."""
    encoder = BehaviorEncoder(
        embedding_dim=768,
        output_dim=256,
        recency_halflife_days=30.0,
    )

    assert encoder.embedding_dim == 768
    assert encoder.output_dim == 256
    assert encoder.recency_halflife_days == 30.0
    assert encoder.projection.shape == (768, 256)


def test_encode_empty_sequence():
    """Test encoding empty purchase sequence."""
    encoder = BehaviorEncoder()

    embedding = encoder.encode_sequence(
        product_embeddings=[],
        timestamps=[],
    )

    assert embedding.shape == (256,)
    assert np.allclose(embedding, 0.0)


def test_encode_single_purchase():
    """Test encoding single purchase."""
    encoder = BehaviorEncoder(embedding_dim=768, output_dim=256)

    # Create a simple product embedding
    product_embedding = np.random.randn(768).astype(np.float32)
    timestamp = datetime.now()

    embedding = encoder.encode_sequence(
        product_embeddings=[product_embedding],
        timestamps=[timestamp],
    )

    assert embedding.shape == (256,)
    # Should be L2 normalized
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_encode_multiple_purchases():
    """Test encoding multiple purchases."""
    encoder = BehaviorEncoder(embedding_dim=768, output_dim=256)

    # Create multiple product embeddings
    product_embeddings = [
        np.random.randn(768).astype(np.float32) for _ in range(5)
    ]
    timestamps = [
        datetime.now() - timedelta(days=i) for i in range(5)
    ]

    embedding = encoder.encode_sequence(
        product_embeddings=product_embeddings,
        timestamps=timestamps,
    )

    assert embedding.shape == (256,)
    # Should be L2 normalized
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_recency_weighting():
    """Test that recent purchases are weighted more."""
    encoder = BehaviorEncoder(
        embedding_dim=768,
        output_dim=256,
        recency_halflife_days=30.0,
    )

    now = datetime.now()

    # Create two distinct product embeddings
    embedding1 = np.random.randn(768).astype(np.float32)
    embedding2 = np.random.randn(768).astype(np.float32)

    # Test 1: Only old purchase
    product_embeddings_old = [embedding1]
    timestamps_old = [now - timedelta(days=90)]
    result_old = encoder.encode_sequence(
        product_embeddings_old, timestamps_old, reference_time=now
    )

    # Test 2: Only recent purchase
    product_embeddings_recent = [embedding2]
    timestamps_recent = [now - timedelta(days=1)]
    result_recent = encoder.encode_sequence(
        product_embeddings_recent, timestamps_recent, reference_time=now
    )

    # Test 3: Both purchases (recent should dominate)
    product_embeddings_both = [embedding1, embedding2]
    timestamps_both = [now - timedelta(days=90), now - timedelta(days=1)]
    result_both = encoder.encode_sequence(
        product_embeddings_both, timestamps_both, reference_time=now
    )

    # Recent purchase should be more similar to the combined result
    # than the old purchase
    similarity_old = np.dot(result_old, result_both)
    similarity_recent = np.dot(result_recent, result_both)

    assert similarity_recent > similarity_old, "Recent purchase should dominate"


def test_recency_weight_calculation():
    """Test recency weight calculation."""
    encoder = BehaviorEncoder(recency_halflife_days=30.0)

    now = datetime.now()
    timestamps = [
        now,                         # Today: weight = 1.0
        now - timedelta(days=30),    # 30 days ago: weight = 0.5
        now - timedelta(days=60),    # 60 days ago: weight = 0.25
        now - timedelta(days=90),    # 90 days ago: weight = 0.125
    ]

    weights = encoder._calculate_recency_weights(timestamps, now)

    assert len(weights) == 4
    assert np.allclose(weights[0], 1.0, atol=1e-5)
    assert np.allclose(weights[1], 0.5, atol=1e-5)
    assert np.allclose(weights[2], 0.25, atol=1e-5)
    assert np.allclose(weights[3], 0.125, atol=1e-5)


def test_encode_batch():
    """Test batch encoding."""
    encoder = BehaviorEncoder(embedding_dim=768, output_dim=256)

    # Create batch of sequences
    sequences = []
    for i in range(3):
        product_embeddings = [
            np.random.randn(768).astype(np.float32) for _ in range(i + 1)
        ]
        timestamps = [
            datetime.now() - timedelta(days=j) for j in range(i + 1)
        ]
        sequences.append((product_embeddings, timestamps))

    embeddings = encoder.encode_batch(sequences)

    assert embeddings.shape == (3, 256)
    # All should be L2 normalized
    for i in range(3):
        if len(sequences[i][0]) > 0:  # Skip empty sequences
            assert np.allclose(np.linalg.norm(embeddings[i]), 1.0, atol=1e-5)


def test_mismatched_lengths_raises_error():
    """Test that mismatched product_embeddings and timestamps raise error."""
    encoder = BehaviorEncoder()

    product_embeddings = [np.random.randn(768).astype(np.float32)]
    timestamps = [datetime.now(), datetime.now()]  # Mismatch

    with pytest.raises(ValueError, match="must have same length"):
        encoder.encode_sequence(product_embeddings, timestamps)


def test_deterministic_encoding():
    """Test that encoding is deterministic."""
    encoder1 = BehaviorEncoder(embedding_dim=768, output_dim=256)
    encoder2 = BehaviorEncoder(embedding_dim=768, output_dim=256)

    product_embeddings = [np.random.randn(768).astype(np.float32) for _ in range(3)]
    timestamps = [datetime.now() - timedelta(days=i) for i in range(3)]
    reference_time = datetime.now()

    embedding1 = encoder1.encode_sequence(
        product_embeddings, timestamps, reference_time
    )
    embedding2 = encoder2.encode_sequence(
        product_embeddings, timestamps, reference_time
    )

    # Should produce identical results (same seed)
    assert np.allclose(embedding1, embedding2)


@pytest.mark.asyncio
async def test_customer_behavior_encoder():
    """Test CustomerBehaviorEncoder interface."""
    encoder = CustomerBehaviorEncoder(
        embedding_dim=768,
        output_dim=256,
        recency_halflife_days=30.0,
    )

    assert encoder.embedding_dim == 256

    # Create test inputs
    inputs = []
    for i in range(2):
        product_embeddings = [
            np.random.randn(768).astype(np.float32) for _ in range(i + 2)
        ]
        timestamps = [
            datetime.now() - timedelta(days=j) for j in range(i + 2)
        ]
        inputs.append((product_embeddings, timestamps))

    embeddings = await encoder.embed(inputs)

    assert len(embeddings) == 2
    assert embeddings[0].shape == (256,)
    assert embeddings[1].shape == (256,)


def test_projection_matrix_normalized():
    """Test that projection matrix columns are normalized."""
    encoder = BehaviorEncoder(embedding_dim=768, output_dim=256)

    # Check each column has unit norm
    for i in range(encoder.projection.shape[1]):
        column_norm = np.linalg.norm(encoder.projection[:, i])
        assert np.allclose(column_norm, 1.0, atol=1e-5)


def test_different_halflife_values():
    """Test encoder with different recency halflife values."""
    encoder_short = BehaviorEncoder(recency_halflife_days=7.0)
    encoder_long = BehaviorEncoder(recency_halflife_days=90.0)

    now = datetime.now()
    timestamps = [now - timedelta(days=30)]

    weights_short = encoder_short._calculate_recency_weights(timestamps, now)
    weights_long = encoder_long._calculate_recency_weights(timestamps, now)

    # Short halflife should decay faster
    assert weights_short[0] < weights_long[0]


def test_output_dtype():
    """Test that output embeddings are float32."""
    encoder = BehaviorEncoder(embedding_dim=768, output_dim=256)

    product_embeddings = [np.random.randn(768).astype(np.float32)]
    timestamps = [datetime.now()]

    embedding = encoder.encode_sequence(product_embeddings, timestamps)

    assert embedding.dtype == np.float32
