"""Exact local YuE2 source/model/numerical candidate; no model download or CUDA work."""
import hashlib
from importlib import metadata
import json
from pathlib import Path

MODEL_ID = 'm-a-p/YuE2-3B'
MODEL_REVISION = '14fc6c6f146441b1dd6363fcb2e01e82a6914cb7'
UPSTREAM_COMMIT = 'ef1936f2ee39fe8de486a0f47a481c95f8d4da87'
UPSTREAM_URL = 'https://github.com/multimodal-art-projection/YuE.git'
PROFILE = 'release-0.9.5'
NUMERICAL_PROFILE = 'yue2-bf16-sdpa-math-strict-pg095-v1'
LOCK = Path(__file__).with_name('native-assets.lock.json')
VERSIONS = {'yue2-infer': '0.1.6', 'torch': '2.10.0',
    'transformers': '4.57.6', 'huggingface-hub': '0.36.2',
    'safetensors': '0.7.0', 'tiktoken': '0.12.0', 'numpy': '2.2.6',
    'soundfile': '0.13.1', 'accelerate': '1.13.0'}
ARCHITECTURE = dict(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8, head_dim=128, vocab_size=184704, intermediate_size=6144,
    max_position_embeddings=24576)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def sources(root):
    root = Path(root)
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def numerical_profile(options, model=None):
    """Inspect caller-owned strict math flags; never changes process settings."""
    import os
    import torch
    if options.get('native_numerical_profile') != NUMERICAL_PROFILE:
        raise ValueError('native YuE2 requires the explicit strict numerical profile')
    actual = dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        math_sdp_low_precision_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    )
    expected = dict(deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_benchmark=False, cudnn_deterministic=True, tf32_matmul=False,
        tf32_cudnn=False, float32_matmul_precision='highest',
        fp16_reduced_precision_reduction=False, bf16_reduced_precision_reduction=False,
        flash_sdp=False, memory_efficient_sdp=False, cudnn_sdp=False, math_sdp=True,
        math_sdp_low_precision_reduction=False, cublas_workspace_config=':4096:8')
    if actual != expected:
        changed = sorted(key for key in expected if actual[key] != expected[key])
        raise ValueError('native YuE2 strict numerical policy differs: ' + ', '.join(changed))
    overrides = {key: os.environ.get(key) for key in
        ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE')}
    if any(value not in (None, '0') for value in overrides.values()):
        raise ValueError('native YuE2 strict numerical policy refuses TF32 environment override')
    implementation = 'sdpa'
    if model is not None and getattr(model.config, '_attn_implementation', None) != implementation:
        raise ValueError('loaded native YuE2 attention implementation differs from strict SDPA')
    return dict(profile=NUMERICAL_PROFILE, actual_flags=actual, environment_overrides=overrides,
        attention=dict(transformers_implementation=implementation,
            native_function='yue2.modeling_yue2.sdpa',
            torch_function='torch.nn.functional.scaled_dot_product_attention',
            enabled_backend='math', native_grouped_query_attention=True))


def verify_native(options):
    """Verify the declared BF16 CUDA candidate before loader allocation/attachment."""
    if (options.get('pg_source_profile'), options.get('penalty_input_domain')) != (PROFILE, 'continuous'):
        raise ValueError('native YuE2 requires its explicit current continuous source/domain pair')
    if options.get('base_precision') != 'bfloat16' or not options.get('device', '').startswith('cuda:'):
        raise ValueError('native YuE2 candidate requires explicit CUDA device and bfloat16 base precision')
    if options.get('revision') != MODEL_REVISION:
        raise ValueError('native YuE2 requires its exact full model revision')
    import torch
    if torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8':
        raise ValueError('native YuE2 requires actual Torch2.10.0+cu128/CUDA12.8')
    numerics = numerical_profile(options)
    actual_versions = {name: metadata.version(name) for name in VERSIONS}
    if actual_versions != VERSIONS:
        raise ValueError('native YuE2 numerical runtime differs from the declared source candidate')
    dist = metadata.distribution('yue2-infer')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    vcs = direct.get('vcs_info', {})
    if (direct.get('url'), vcs.get('vcs'), vcs.get('commit_id')) != (UPSTREAM_URL, 'git', UPSTREAM_COMMIT) or direct.get('dir_info', {}).get('editable') or 'archive_info' in direct:
        raise ValueError('native YuE2 requires exact immutable upstream Git origin')
    lock = json.loads(LOCK.read_text())
    if (lock.get('schema'), lock.get('model_id'), lock.get('revision'), lock.get('upstream_commit'), lock.get('architecture')) != (
            'yue2-native-assets-v1', MODEL_ID, MODEL_REVISION, UPSTREAM_COMMIT, ARCHITECTURE):
        raise ValueError('native YuE2 asset descriptor differs from the source cohort')
    import importlib.util, sys
    upstream_root = Path(dist.locate_file('yue2')).resolve()
    spec = importlib.util.find_spec('yue2')
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != upstream_root / '__init__.py':
        raise ValueError('native YuE2 upstream package import origin differs')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] == 'yue2':
            origin = getattr(module, '__file__', None)
            if origin is None or not Path(origin).resolve().is_relative_to(upstream_root):
                raise ValueError('native YuE2 upstream module import origin differs')
    if sources(upstream_root) != lock['upstream_sources_sha256']:
        raise ValueError('native YuE2 installed upstream source closure differs')
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
        cache_dir=options.get('cache_dir'), local_files_only=True))
    if sources(model) != lock['model_sources_sha256']:
        raise ValueError('native YuE2 complete cached model/tokenizer bytes differ')
    config = json.loads((model / 'config.json').read_text())
    if {key: config.get(key) for key in ARCHITECTURE} != ARCHITECTURE or config.get('dtype', config.get('torch_dtype')) != 'bfloat16':
        raise ValueError('native YuE2 cached architecture/precision differs')
    return dict(schema='yue2-native-assets-v1', model_id=MODEL_ID, revision=MODEL_REVISION,
        manifest_sha256=file_sha256(LOCK), upstream_commit=UPSTREAM_COMMIT,
        runtime_versions=actual_versions, actual_torch=torch.__version__, cuda_build=torch.version.cuda, architecture=ARCHITECTURE, numerical_profile=numerics,
        qualification='exact native BF16 source candidate; native GPU/resume/acoustic quality not yet qualified')


def verify_loaded_model(model, options):
    """Compare every loaded frozen tensor to its checkpoint before any attachment."""
    import torch
    from safetensors import safe_open
    if {key: getattr(model.config, key, None) for key in ARCHITECTURE} != ARCHITECTURE:
        raise ValueError('loaded native YuE2 architecture differs from its exact checkpoint')
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
        cache_dir=options.get('cache_dir'), local_files_only=True))
    state = model.state_dict()
    digest = hashlib.sha256()
    with safe_open(path / 'model.safetensors', framework='pt', device='cpu') as checkpoint:
        if set(state) != set(checkpoint.keys()):
            raise ValueError('loaded native YuE2 checkpoint key inventory differs')
        for name, value in state.items():
            expected = checkpoint.get_tensor(name)
            if str(value.device) != options['device'] or value.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
                raise ValueError('loaded native YuE2 checkpoint device/precision differs: ' + name)
            actual = value.detach().cpu().contiguous()
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError('loaded native YuE2 checkpoint tensor differs: ' + name)
            digest.update(name.encode())
            digest.update(actual.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(complete_checkpoint_tensors=len(state), all_values_exact=True,
        all_checkpoint_dtypes='bfloat16', frozen_base_sha256=digest.hexdigest())
