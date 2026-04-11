"""
Tests unitarios para los 5 bloques de métricas de evaluación geométrica.
"""

import torch
import pytest

from geoquant.evaluation import block_a, block_b, block_c, block_d, block_e


class TestBlockA:
    def test_cosine_drift_identical(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        drift = block_a.cosine_drift(emb, emb)
        assert drift == pytest.approx(0.0, abs=1e-5), "Drift debe ser 0 para embeddings idénticos"

    def test_cosine_drift_range(self, dummy_embeddings_pair):
        emb_fp32, emb_int8, _ = dummy_embeddings_pair
        drift = block_a.cosine_drift(emb_fp32, emb_int8)
        assert 0.0 <= drift <= 2.0

    def test_rre_identical(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        assert block_a.rre(emb, emb) == pytest.approx(0.0, abs=1e-5)

    def test_rre_positive(self, dummy_embeddings_pair):
        emb_fp32, emb_int8, _ = dummy_embeddings_pair
        assert block_a.rre(emb_fp32, emb_int8) >= 0.0

    def test_run_keys(self, dummy_embeddings_pair):
        emb_fp32, emb_int8, _ = dummy_embeddings_pair
        result = block_a.run(emb_fp32, emb_int8)
        assert "cosine_drift" in result
        assert "rre" in result


class TestBlockB:
    def test_cka_identical(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        assert block_b.cka_linear(emb, emb) == pytest.approx(1.0, abs=1e-4)

    def test_cka_range(self, dummy_embeddings_pair):
        emb_fp32, emb_int8, _ = dummy_embeddings_pair
        cka = block_b.cka_linear(emb_fp32, emb_int8)
        assert 0.0 <= cka <= 1.0

    def test_uniformity_negative(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        # La uniformity es log de una exponencial media, generalmente negativa
        u = block_b.uniformity(emb)
        assert isinstance(u, float)

    def test_alignment_positive(self, dummy_embeddings):
        emb, labels = dummy_embeddings
        a = block_b.alignment(emb, labels)
        assert a >= 0.0


class TestBlockC:
    def test_overlap_identical(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        overlap = block_c.overlap_at_k(emb, emb, k=5)
        assert overlap == pytest.approx(1.0, abs=1e-5)

    def test_trustworthiness_range(self, dummy_embeddings_pair):
        emb_fp32, emb_int8, _ = dummy_embeddings_pair
        tw = block_c.trustworthiness_score(emb_fp32, emb_int8, k=5)
        assert 0.0 <= tw <= 1.0

    def test_run_keys(self, dummy_embeddings_pair):
        emb_fp32, emb_int8, _ = dummy_embeddings_pair
        result = block_c.run(emb_fp32, emb_int8, k=5)
        assert "overlap_at_5" in result
        assert "trustworthiness" in result


class TestBlockD:
    def test_iso_mean_identical(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        assert block_d.iso_mean(emb, emb, sample=50) == pytest.approx(0.0, abs=1e-4)

    def test_edim_positive(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        edim = block_d.effective_dim(emb)
        assert edim >= 1.0

    def test_edim_at_most_d(self, dummy_embeddings):
        emb, _ = dummy_embeddings
        edim = block_d.effective_dim(emb)
        assert edim <= emb.shape[1]


class TestBlockE:
    def test_knn_accuracy_range(self, dummy_embeddings):
        emb, labels = dummy_embeddings
        acc = block_e.knn_accuracy(emb, labels, k=3)
        assert 0.0 <= acc <= 1.0

    def test_knn_accuracy_type(self, dummy_embeddings):
        emb, labels = dummy_embeddings
        acc = block_e.knn_accuracy(emb, labels, k=3)
        assert isinstance(acc, float)

    def test_linear_probe_range(self, dummy_embeddings):
        emb, labels = dummy_embeddings
        acc = block_e.linear_probe(emb, labels, emb, labels, max_iter=100)
        assert 0.0 <= acc <= 1.0