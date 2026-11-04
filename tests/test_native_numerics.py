"""Caller-owned strict math policy: no guard may mutate flags or initialize CUDA."""
from types import SimpleNamespace

import pytest
import torch

from yue2_sliders.native_assets import numerical_profile, NUMERICAL_PROFILE


@pytest.fixture
def strict_math(monkeypatch):
    old_precision=torch.get_float32_matmul_precision()
    old_deterministic=torch.are_deterministic_algorithms_enabled()
    old_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    settings=[(torch.backends.cudnn,'benchmark',False),(torch.backends.cudnn,'deterministic',True),
        (torch.backends.cudnn,'allow_tf32',False),(torch.backends.cuda.matmul,'allow_tf32',False),
        (torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction',False),
        (torch.backends.cuda.matmul,'allow_bf16_reduced_precision_reduction',False)]
    originals=[(owner,key,getattr(owner,key)) for owner,key,_ in settings]
    backends=[(torch.backends.cuda.enable_flash_sdp,torch.backends.cuda.flash_sdp_enabled,False),
        (torch.backends.cuda.enable_mem_efficient_sdp,torch.backends.cuda.mem_efficient_sdp_enabled,False),
        (torch.backends.cuda.enable_cudnn_sdp,torch.backends.cuda.cudnn_sdp_enabled,False),
        (torch.backends.cuda.enable_math_sdp,torch.backends.cuda.math_sdp_enabled,True),
        (torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp,torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed,False)]
    old_backends=[(setter,getter()) for setter,getter,_ in backends]
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True,warn_only=False)
    for owner,key,value in settings:setattr(owner,key,value)
    for setter,_,value in backends:setter(value)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',raising=False)
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE',raising=False)
    try:yield dict(native_numerical_profile=NUMERICAL_PROFILE)
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.use_deterministic_algorithms(old_deterministic,warn_only=old_warn)
        for owner,key,value in originals:setattr(owner,key,value)
        for setter,value in old_backends:setter(value)


def test_native_math_identity_records_actual_flags_and_model_attention(strict_math):
    native=SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa'))
    identity=numerical_profile(strict_math,native)
    assert identity['profile']==NUMERICAL_PROFILE
    assert identity['actual_flags']['math_sdp']
    assert not identity['actual_flags']['flash_sdp']
    assert not identity['actual_flags']['tf32_matmul']
    assert identity['attention']['enabled_backend']=='math'
    assert identity['attention']['transformers_implementation']=='sdpa'
    assert numerical_profile(strict_math,native)==identity  # Inspection changes no state.
    native.config._attn_implementation='eager'
    with pytest.raises(ValueError,match='attention implementation'):
        numerical_profile(strict_math,native)
    assert not torch.cuda.is_initialized()


def test_native_profile_must_be_explicit_even_when_flags_match(strict_math):
    with pytest.raises(ValueError,match='explicit strict numerical profile'):
        numerical_profile({})


@pytest.mark.parametrize('setting', ['deterministic','warn-only','cudnn-benchmark','cudnn-deterministic',
    'tf32-matmul','tf32-cudnn','fp16-reduction','bf16-reduction','flash','efficient','cudnn-sdp',
    'math-disabled','math-low-precision','workspace','tf32-env','nvidia-tf32-env'])
def test_changed_math_policy_refuses_without_changing_it(strict_math,monkeypatch,setting):
    if setting=='deterministic':torch.use_deterministic_algorithms(False)
    elif setting=='warn-only':torch.use_deterministic_algorithms(True,warn_only=True)
    elif setting=='cudnn-benchmark':torch.backends.cudnn.benchmark=True
    elif setting=='cudnn-deterministic':torch.backends.cudnn.deterministic=False
    elif setting=='tf32-matmul':torch.backends.cuda.matmul.allow_tf32=True
    elif setting=='tf32-cudnn':torch.backends.cudnn.allow_tf32=True
    elif setting=='fp16-reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True
    elif setting=='bf16-reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    elif setting=='flash':torch.backends.cuda.enable_flash_sdp(True)
    elif setting=='efficient':torch.backends.cuda.enable_mem_efficient_sdp(True)
    elif setting=='cudnn-sdp':torch.backends.cuda.enable_cudnn_sdp(True)
    elif setting=='math-disabled':torch.backends.cuda.enable_math_sdp(False)
    elif setting=='math-low-precision':torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    elif setting=='workspace':monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':16:8')
    elif setting=='tf32-env':monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','1')
    elif setting=='nvidia-tf32-env':monkeypatch.setenv('NVIDIA_TF32_OVERRIDE','1')
    with pytest.raises(ValueError,match='strict numerical policy'):
        numerical_profile(strict_math)
    assert not torch.cuda.is_initialized()


def test_pinned_native_loader_accepts_explicit_sdpa_on_small_real_class(tmp_path,monkeypatch,strict_math):
    native=pytest.importorskip('yue2.modeling_yue2')
    storage=pytest.importorskip('yue2.storage')
    tokenizer=pytest.importorskip('yue2.tokenization_yue2')
    from yue2_sliders.backend import YuE2Backend
    config=native.YuE2Config(hidden_size=32,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=16,intermediate_size=64,vocab_size=256,
        max_position_embeddings=64,max_latent_frames=32,latent_dim=8)
    model=native.YuE2ForCausalLM(config)
    model.save_pretrained(tmp_path)
    monkeypatch.setattr(storage,'resolve_model',lambda *a,**k:tmp_path)
    monkeypatch.setattr(tokenizer,'YuE2TextTokenizer',lambda *a,**k:None)
    monkeypatch.setattr(storage,'model_identity',lambda *a,**k:{'fixture':'small actual upstream class; no pretrained qualification'})
    loaded=YuE2Backend(device='cpu',attention_implementation='sdpa')
    assert loaded.model.config._attn_implementation=='sdpa'
    assert numerical_profile(strict_math,loaded.model)['attention']['enabled_backend']=='math'
    with torch.no_grad():response=loaded.hidden([1,2,3])
    assert response.shape==(1,3,32) and torch.isfinite(response).all()
    assert not torch.cuda.is_initialized()


def test_changed_policy_refuses_before_native_loader_allocation(monkeypatch,strict_math):
    from particle_sliders.pg09 import _released_sources
    try:_released_sources('source-801ae1e-continuous-v1')
    except RuntimeError:pytest.skip('separate exact PG801 candidate environment required')
    from yue2_sliders import backend
    from yue2_sliders.response_native import build_native
    from yue2_sliders.native_assets import MODEL_REVISION
    def forbidden(*args,**kwargs):raise AssertionError('native constructor reached before policy refusal')
    monkeypatch.setattr(backend,'YuE2Backend',forbidden)
    torch.backends.cuda.enable_flash_sdp(True)
    with pytest.raises(ValueError,match='strict numerical policy.*flash_sdp'):
        build_native(dict(strict_math,device='cuda:0',base_precision='bfloat16',
            revision=MODEL_REVISION,pg_source_profile='source-801ae1e-continuous-v1',
            penalty_input_domain='continuous'))
    assert not torch.cuda.is_initialized()


def test_numerical_observation_refuses_changed_policy_before_native_body(strict_math):
    from yue2_sliders.response_runtime import observation
    model=torch.nn.Linear(2,2).eval().requires_grad_(False)
    model.config=SimpleNamespace(_attn_implementation='sdpa')
    runtime=SimpleNamespace(model=model)
    identity=numerical_profile(strict_math,model)
    with observation(runtime,native_options=strict_math,expected_numerics=identity):
        pass
    torch.backends.cuda.enable_flash_sdp(True)
    reached=False
    with pytest.raises(ValueError,match='strict numerical policy'):
        with observation(runtime,native_options=strict_math,expected_numerics=identity):
            reached=True
    assert not reached and not torch.cuda.is_initialized()
