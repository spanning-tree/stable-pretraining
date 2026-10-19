"""Unit tests for the ResidualPred predictor options (identity attention bias, zero-init MLP)."""

import pytest
import torch

from stable_pretraining.backbone import FlexibleTransformer
from stable_pretraining.methods.ijepa import IJEPA
from stable_pretraining.optim.utils import split_params_for_weight_decay

pytestmark = pytest.mark.unit

B, N_CTX, N_QRY, DIM = 2, 12, 4, 32
NUM_PATCHES = 16


def make_predictor(**kwargs):
    defaults = dict(
        input_dim=DIM,
        hidden_dim=DIM,
        output_dim=DIM,
        num_patches=NUM_PATCHES,
        depth=2,
        num_heads=4,
        self_attn=True,
        cross_attn=False,
        use_adaln=False,
        add_mask_token=True,
        num_prefix_tokens=0,
        zero_init_output=False,
    )
    defaults.update(kwargs)
    return FlexibleTransformer(**defaults)


def make_inputs():
    perm = torch.randperm(NUM_PATCHES)
    context_idx = perm[:N_CTX].expand(B, -1)
    query_idx = perm[N_CTX : N_CTX + N_QRY].expand(B, -1)
    context = torch.randn(B, N_CTX, DIM)
    queries = torch.zeros(B, N_QRY, DIM)
    query_mask = torch.ones(B, N_QRY, dtype=torch.bool)
    return context, queries, context_idx, query_idx, query_mask


class TestIdentityAttnBias:
    """FlexibleTransformer's identity_attn_bias option."""

    def test_default_has_no_bias(self):
        assert make_predictor().identity_attn_bias is None

    def test_one_bias_per_block_and_head(self):
        predictor = make_predictor(identity_attn_bias=10.0)
        assert len(predictor.identity_attn_bias) == 2
        for beta in predictor.identity_attn_bias:
            assert beta.shape == (4,)
            assert torch.all(beta == 10.0)

    def test_zero_bias_matches_no_bias(self):
        torch.manual_seed(0)
        plain = make_predictor()
        biased = make_predictor(identity_attn_bias=0.0)
        biased.load_state_dict(plain.state_dict(), strict=False)
        plain.eval()
        biased.eval()

        context, queries, context_idx, query_idx, query_mask = make_inputs()
        args = (context, queries, context_idx, query_idx)
        out_plain = plain(*args, query_mask=query_mask)
        out_biased = biased(*args, query_mask=query_mask)
        torch.testing.assert_close(out_plain, out_biased)

    def test_large_bias_isolates_context_tokens(self):
        """With a large bias each context token attends (almost) only to itself."""
        torch.manual_seed(0)
        predictor = make_predictor(identity_attn_bias=50.0).eval()
        context, queries, context_idx, query_idx, query_mask = make_inputs()

        out = predictor(
            context,
            queries,
            context_idx,
            query_idx,
            query_mask=query_mask,
            return_all=True,
        )
        perturbed = context.clone()
        perturbed[:, 1:] += torch.randn_like(perturbed[:, 1:])
        out_perturbed = predictor(
            perturbed,
            queries,
            context_idx,
            query_idx,
            query_mask=query_mask,
            return_all=True,
        )

        first = context_idx[0, 0]
        torch.testing.assert_close(
            out[:, first], out_perturbed[:, first], atol=1e-4, rtol=1e-4
        )

    def test_queries_still_read_context(self):
        """Query tokens get no bias, so they keep attending to the context."""
        torch.manual_seed(0)
        predictor = make_predictor(identity_attn_bias=50.0).eval()
        context, queries, context_idx, query_idx, query_mask = make_inputs()

        out = predictor(context, queries, context_idx, query_idx, query_mask=query_mask)
        out_perturbed = predictor(
            context + torch.randn_like(context),
            queries,
            context_idx,
            query_idx,
            query_mask=query_mask,
        )
        assert not torch.allclose(out, out_perturbed, atol=1e-3)

    def test_bias_receives_gradient(self):
        predictor = make_predictor(identity_attn_bias=1.0)
        context, queries, context_idx, query_idx, query_mask = make_inputs()
        out = predictor(
            context,
            queries,
            context_idx,
            query_idx,
            query_mask=query_mask,
            return_all=True,
        )
        out.pow(2).mean().backward()
        for beta in predictor.identity_attn_bias:
            assert beta.grad is not None
            assert torch.isfinite(beta.grad).all()
            assert beta.grad.abs().sum() > 0

    def test_last_block_bias_has_no_gradient_from_queries_only(self):
        """The last block's bias only changes context outputs, which a query-only loss ignores."""
        predictor = make_predictor(identity_attn_bias=1.0)
        context, queries, context_idx, query_idx, query_mask = make_inputs()
        out = predictor(context, queries, context_idx, query_idx, query_mask=query_mask)
        out.pow(2).mean().backward()
        assert predictor.identity_attn_bias[0].grad.abs().sum() > 0
        assert torch.all(predictor.identity_attn_bias[-1].grad == 0)

    def test_combines_with_boolean_attn_mask(self):
        predictor = make_predictor(identity_attn_bias=10.0).eval()
        context, queries, context_idx, query_idx, query_mask = make_inputs()
        leave_one_out = torch.eye(N_CTX, dtype=torch.bool)
        out = predictor(
            context,
            queries,
            context_idx,
            query_idx,
            query_mask=query_mask,
            attn_mask=leave_one_out,
        )
        assert torch.isfinite(out).all()

    def test_bias_excluded_from_weight_decay(self):
        predictor = make_predictor(identity_attn_bias=10.0)
        groups = split_params_for_weight_decay(
            predictor.named_parameters(), weight_decay=0.05
        )
        no_decay = next(g for g in groups if g["weight_decay"] == 0.0)
        no_decay_ids = {id(p) for p in no_decay["params"]}
        for beta in predictor.identity_attn_bias:
            assert id(beta) in no_decay_ids

    def test_rejects_cross_attention(self):
        with pytest.raises(ValueError, match="identity_attn_bias"):
            make_predictor(cross_attn=True, identity_attn_bias=10.0)


class TestZeroInitMlp:
    """FlexibleTransformer's zero_init_mlp option."""

    @pytest.mark.parametrize("mlp_type", ["gelu", "swiglu"])
    def test_mlp_output_is_zero(self, mlp_type):
        predictor = make_predictor(zero_init_mlp=True, mlp_type=mlp_type)
        for block in predictor.blocks:
            out_layer = block.mlp.w3 if mlp_type == "swiglu" else block.mlp.fc2
            assert torch.all(out_layer.weight == 0)
            if out_layer.bias is not None:
                assert torch.all(out_layer.bias == 0)

    def test_default_mlp_is_not_zero(self):
        predictor = make_predictor()
        assert predictor.blocks[0].mlp.fc2.weight.abs().sum() > 0


class TestIJEPAResidualPred:
    """IJEPA with the ResidualPred predictor."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return IJEPA(
            "vit_tiny_patch16_224",
            predictor_embed_dim=64,
            predictor_depth=2,
            predictor_identity_bias=10.0,
            predictor_zero_init_mlp=True,
        )

    def test_predictor_configured(self, model):
        assert model.predictor.identity_attn_bias is not None
        assert torch.all(model.predictor.blocks[0].mlp.fc2.weight == 0)

    def test_train_step(self, model):
        model.train()
        output = model(torch.randn(2, 3, 224, 224))
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert model.predictor.identity_attn_bias[0].grad is not None

    def test_default_ijepa_unchanged(self):
        model = IJEPA("vit_tiny_patch16_224", predictor_embed_dim=64, predictor_depth=2)
        assert model.predictor.identity_attn_bias is None
        assert model.predictor.blocks[0].mlp.fc2.weight.abs().sum() > 0
