"""Explicit fresh/FULL/RAW owner for original YuE2 final AR responses, not PCM."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib,torch
from particle_sliders.pg095 import ConditionalPG095
from . import PROFILE,ownership,policy
from .cache import load_cache,to_device,teacher_identity
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json
FULL_SCHEMA='YuE2-explicit-cached-fresh95-FULL-v1'
RAW_SCHEMA='YuE2-explicit-cached-fresh95-RAW-v1'
CONSTRUCTOR='new-owner-original-ParticleSlider-seed-and-critic-seed-plus1-v1'

def original_options(config):
    value={k:config[k]for k in('device','base_precision','pg_source_profile','penalty_input_domain')}
    value.update(dummy=config['lane']=='cpu-fixture',base_seed=config['seed'])
    if config['lane']=='native2048':value.update({k:config[k]for k in('revision','cache_dir','native_numerical_profile')})
    return value

def configuration(value):
    if not isinstance(value,dict):raise ValueError('explicit YuE2 configuration required')
    common={'lane','model_identity','steps','batch','seed','width','device','base_precision','checkpointing','max_sequence_length','pg_source_profile','penalty_input_domain','preset','ema_decay','lr','d_lr_mult','prior_lr_mult'}
    native=value.get('lane')=='native2048'
    if value.get('lane')not in('cpu-fixture','native2048')or set(value)!=(common|({'revision','cache_dir','native_numerical_profile'}if native else set())):raise ValueError('exact explicit original CPU stand-in or native2048 options required')
    if value['ema_decay']is not None:raise ValueError('weight EMA refused; historical EMA0.9 cohort is not this fresh profile')
    if not isinstance(value['model_identity'],dict)or not value['model_identity']:raise ValueError('exact original runtime/base identity required')
    if(not native and(value['width']!=32 or value['device']!='cpu'or value['base_precision']!='float32'or value['checkpointing']is not False)or native and value['width']!=2048):raise ValueError('original32-wide CPU fixture or full2048 native response required')
    if type(value['seed'])is not int or not 0<=value['seed']<2**63-100 or type(value['steps'])is not int or not 1<=value['steps']<=1000000 or type(value['batch'])is not int or not 2<=value['batch']<=512:raise ValueError('bounded original seed/steps/batch required')
    if type(value['max_sequence_length'])is not int or not 1<=value['max_sequence_length']<=(24576 if native else 256):raise ValueError('original whole context limit required')
    if type(value['checkpointing'])is not bool or value['preset']not in('bispf','bispf_optimized'):raise ValueError('original checkpoint/preset required')
    from yue2_sliders.response_runtime import validate_native_options,capabilities
    from yue2_sliders.learning_rates import recipe_overrides
    options=original_options(value);validate_native_options(options)
    if native:
        from yue2_sliders.native_assets import verify_native
        verify_native(options)  # Exact unchanged original native runtime/assets/numerics gate, no fallback.
    if any(value[k]is None for k in('lr','d_lr_mult','prior_lr_mult')):raise ValueError('explicit owner rates; no old preset migration')
    recipe_overrides(**{k:value[k]for k in('lr','d_lr_mult','prior_lr_mult')});capabilities(options).check('train',precision=value['base_precision'])
    canonical(value);return deepcopy(value)

def intent(mode,resume,cohort,ema_decay):
    if mode not in('fresh','resume')or not isinstance(cohort,str)or not cohort.strip():raise ValueError('explicit fresh/resume/cohort required')
    if mode=='fresh'and resume is not None or mode=='resume'and resume is None:raise ValueError('fresh refuses state; exact new FULL required for strict resume')
    if ema_decay is not None:raise ValueError('weight EMA refused')

def model_identity(runtime,config):
    from yue2_sliders.backend import YuE2Backend,TinyYuE2,architecture_of
    from yue2_sliders.response_runtime import validate_staged,verify_runtime,native_sources,frozen_digest
    if type(runtime)is not YuE2Backend:raise ValueError('original genuine YuE2Backend owner component required')
    model=runtime.model;options=original_options(config);validate_staged(runtime,options)
    if int(model.config.max_position_embeddings)!=config['max_sequence_length']:raise ValueError('whole original context; no truncation')
    if any('forward'in vars(m)or m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('original pristine native host required')
    loaded=numerics=assets=None
    if config['lane']=='cpu-fixture':
        if type(model)is not TinyYuE2 or architecture_of(model)!=dict(hidden_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,vocab_size=256,intermediate_size=64):raise ValueError('exact original CPU stand-in geometry required')
    else:
        from yue2_sliders.backend import require_yue2
        from yue2_sliders.native_assets import verify_loaded_model,verify_native,numerical_profile
        if type(model)is not require_yue2()[1]or 'hidden'in vars(runtime):raise ValueError('exact original native class/hidden route required')
        assets=verify_native(options);numerics=numerical_profile(options,model);loaded=verify_loaded_model(model,options)
    # Explicit NEW owner constructor stream is in options.CONSTRUCTOR; this is base/Source identity only.
    return dict(base=deepcopy(runtime.identity),frozen_base_sha256=loaded['frozen_base_sha256']if loaded else frozen_digest(model),loaded_checkpoint=loaded,
        native_numerical_profile=numerics,dependencies=verify_runtime(options),native_assets=assets,product_sources=native_sources(),base_seed=config['seed'])

class ReceiptBackend:
    def __init__(self,adapter,receipt,sources,cache):self.backend=adapter;self.network=adapter.network;self.receipt=deepcopy(receipt);self.sources=deepcopy(sources);self.cache=deepcopy(cache)
    def __getattr__(self,name):return getattr(self.backend,name)
    def metadata(self):return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,receipt=deepcopy(self.receipt),sources=deepcopy(self.sources)),completed_native_cache=deepcopy(self.cache),teacher_collector_called=False)

class NativeOwner(ownership.Owner):
    def _components(self,request,factory):
        self.runtime=factory(deepcopy(request));self.model=self.runtime.model
        if model_identity(self.runtime,self.config)!=self.config['model_identity']:raise ValueError('actual original native/base identity differs')
        self.attachment=ownership.Attachment(self.runtime);self.adapter=self.attachment.construct(self.config);self.slider=self.attachment.slider;self.network=self.adapter.network
        if self.network.particles is not self.slider.particles:raise RuntimeError('original bank Parameter identity lost')
    @contextmanager
    def observation(self,seed):
        from particle_sliders.lifecycle import _network_observation
        from yue2_sliders.response_runtime import observation
        self.require_open();device=torch.device(self.config['device']);devices=[device.index]if device.type=='cuda'else[]
        with _network_observation(self.network,seed,devices=devices),observation(self.runtime,native_options=None if self.config['lane']=='cpu-fixture'else original_options(self.config),expected_numerics=self.config['model_identity']['native_numerical_profile']):yield
    def response(self,*,token_ids,strength=1.,seed=7):
        from yue2_sliders.response_runtime import capabilities
        from yue2_sliders.response_native import check_tokens
        from yue2_sliders.backend import architecture_of
        capabilities(original_options(self.config)).check('response',strength=strength)
        check_tokens(token_ids,architecture_of(self.model),self.config['max_sequence_length'])
        with self.observation(seed),torch.no_grad():return self.adapter.response(token_ids=token_ids,strength=strength,seed=seed)

class Training(NativeOwner):
    def __init__(self,config,cache,directory,*,mode,cohort_id,component_factory,owner_admission,resume=None,ema_decay=None):
        super().__init__();intent(mode,resume,cohort_id,ema_decay)
        if not callable(component_factory)or not callable(owner_admission):raise ValueError('genuine component factory/admitting owner required')
        self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable directory required')
        self.runtime_identity=ownership.guard();self.config=configuration(config);self.options=dict(config=self.config,constructor=CONSTRUCTOR)
        self.sources=ownership.sources();self.cohort_id=cohort_id;self.construction=policy.Construction(mode,cohort_id,hashlib.sha256(canonical(self.options).encode()).hexdigest())
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','game','ema_decay'}
            if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA or packet['profile']!=PROFILE or packet['cohort_id']!=cohort_id
                or packet['options']!=self.options or packet['sources']!=self.sources or packet['ema_decay']is not None):raise ValueError('exact whole new profile FULL required; no old cohort reset/migration')
            policy.validate_receipt(packet['initialization'],self.construction,self.options,self.sources)
        prepared,conditions,self.cache_identity=load_cache(cache,config=self.config)
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('same completed original cache required')
        request=dict(operation=mode,profile=PROFILE,cohort_id=cohort_id,options=deepcopy(self.options),completed_cache=deepcopy(self.cache_identity),new_teacher_or_tokenizer_calls=0,fresh_PG_initialization_calls=int(mode=='fresh'),ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not authenticate/admit exact operation')
        try:
            self._components(request,component_factory)
            from yue2_sliders.response_native import native_initialization_rng
            from particle_sliders.pg09_audio import AudioResponseCritic
            options=original_options(self.config);options['base_seed']+=1
            with native_initialization_rng(options):self.critic=AudioResponseCritic(self.config['width'],conditions.shape[1]).to(self.config['device'])
            self.initialization=policy.initialize(self.slider,self.critic,self.construction,self.options,self.sources)if mode=='fresh'else deepcopy(packet['initialization'])
            new_directory(self.directory);write_json(self.directory/'initialization.json',self.initialization)
            self.backend=ReceiptBackend(self.adapter,self.initialization,self.sources,self.cache_identity)
            self.game=ConditionalPG095(self.backend,to_device(prepared,self.config['device']),self.critic,conditions=conditions.to(self.config['device']),particles=self.slider.particles,
                response_domain='continuous_vector',response_scale=1.0,preset=self.config['preset'],total_steps=self.config['steps'],batch_size=self.config['batch'],seed=self.config['seed'],
                pg_source_profile=self.config['pg_source_profile'],penalty_input_domain=self.config['penalty_input_domain'],recipe_overrides={k:self.config[k]for k in('lr','d_lr_mult','prior_lr_mult')})
            if packet is not None:
                self.game.load_state_dict(packet['game'])
                if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('strict whole game/optimizers/cursors/RNG state differs')
            self.inherited_steps=self.game.completed_steps
            self.entry_FULL=self.save(self.directory/f'FULL-{self.inherited_steps:06d}.pt');self.entry_RAW=self.export(self.directory/f'RAW-{self.inherited_steps:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('YuE2 entry and cleanup failed',[primary,cleanup])
            raise
    def update(self):
        self.require_open()
        if self.config['lane']=='native2048':
            from yue2_sliders.native_assets import numerical_profile
            if numerical_profile(original_options(self.config),self.model)!=self.config['model_identity']['native_numerical_profile']:raise ValueError('native numerical policy changed')
        return self.game.step()
    def save(self,path):
        self.require_open();return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open();state=self.game.export_state_dict();return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),cache_identity=deepcopy(self.cache_identity),network=state['network'],completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))

class Inference(NativeOwner):
    def __init__(self,raw,*,component_factory,owner_admission):
        super().__init__();packet=read_packet(raw)
        keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','network','completed_steps','selection','ema_decay'}
        if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=RAW_SCHEMA or packet['profile']!=PROFILE or packet['selection']!='raw'or packet['ema_decay']is not None
            or type(packet['completed_steps'])is not int or packet['completed_steps']<0 or not callable(component_factory)or not callable(owner_admission)):raise ValueError('exact typed new RAW with actual owner required')
        self.runtime_identity=ownership.guard();self.config=configuration(packet['options']['config']);self.options=dict(config=self.config,constructor=CONSTRUCTOR);self.sources=ownership.sources()
        if(packet['options']!=self.options or packet['sources']!=self.sources or packet['cache_identity']['teacher_identity']!=teacher_identity(self.config)or packet['completed_steps']>self.config['steps']):raise ValueError('exact source/lane/config/cache/update identity required')
        construction=policy.Construction('inference',packet['cohort_id'],hashlib.sha256(canonical(self.options).encode()).hexdigest());policy.validate_receipt(packet['initialization'],construction,self.options,self.sources)
        self.initialization=deepcopy(packet['initialization'])
        request=dict(operation='inference',profile=PROFILE,cohort_id=packet['cohort_id'],options=deepcopy(self.options),completed_cache_metadata_only=deepcopy(packet['cache_identity']),new_teacher_or_tokenizer_calls=0,fresh_PG_initialization_calls=0,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not authenticate/admit exact native RAW operation')
        try:
            self._components(request,component_factory);self.network.load_state_dict(packet['network'],strict=True)
            if not equal(self.network.state_dict(),packet['network']):raise RuntimeError('strict RAW addednetwork state differs')
            self.completed_steps=packet['completed_steps']
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('YuE2 RAW and cleanup failed',[primary,cleanup])
            raise

def open_training(*,config,cache,directory,mode,cohort_id,component_factory,owner_admission,resume=None,ema_decay=None):return Training(config,cache,directory,mode=mode,cohort_id=cohort_id,component_factory=component_factory,owner_admission=owner_admission,resume=resume,ema_decay=ema_decay)
def open_inference(raw,*,component_factory,owner_admission):return Inference(raw,component_factory=component_factory,owner_admission=owner_admission)
