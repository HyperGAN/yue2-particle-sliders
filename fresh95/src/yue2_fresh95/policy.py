"""Official PG095 same-Parameter QR at native a=1 RMS; bank R2Normal std1."""
from dataclasses import dataclass
from importlib import metadata
import hashlib,math,torch
from torch import nn
from particlegan import init as pg_init
from . import PROFILE
from .util import canonical,source_record
INIT_SHA='3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b'
SCHEMA='YuE2-native-a1-downQR-std1-R2bank-zeroUP-othersKEEP-initialization-v1'
MARKER='_yue2_fresh95_initialization_receipt'

@dataclass(frozen=True)
class Construction:
    mode:str
    cohort:str
    options_sha256:str

class DeclaredOwned(nn.Module):
    def __init__(self,rows):
        super().__init__();self.specs={}
        for i,row in enumerate(rows):
            key='p'+str(i);self.register_parameter(key,row['parameter']);self.specs[key]=row['distribution']
pg_init.register(DeclaredOwned,lambda value:value.specs)

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def seed_for(seed,name):return int.from_bytes(hashlib.sha256(f'{PROFILE}:{seed}:{name}'.encode()).digest()[:8],'big')%2**63

def declarations(slider,critic):
    from yue2_sliders.slider import ParticleSlider,RoutedProjection
    from particle_sliders.pg09_audio import AudioResponseCritic
    if type(slider)is not ParticleSlider or type(critic)is not AudioResponseCritic or slider.rank!=8 or slider.alpha!=8.:raise ValueError('exact original ParticleSlider rank8/alpha8 and critic required')
    if tuple(slider.particles.shape)!=(128,4)or not slider.adapters:raise ValueError('same original128x4 bank and projections required')
    down={};up={}
    for lora in slider.adapters.values():
        if type(lora)is not RoutedProjection or lora.rank!=8 or lora.scale!=1.:raise ValueError('original RoutedProjection required')
        a,b=lora.lora_down,lora.lora_up
        if(type(a)is not nn.Linear or type(b)is not nn.Linear or a.bias is not None or b.bias is not None or a.out_features!=8 or b.in_features!=8 or torch.count_nonzero(b.weight).item()):raise ValueError('original linear down and genuine fresh zero-UP required')
        down[id(a.weight)]=a.in_features;up[id(b.weight)]=True
    rows=[]
    for name,p in slider.named_parameters(remove_duplicate=False):
        if p is slider.particles:distribution,role=pg_init.R2Normal(0.,1.),'same-original-bank-R2Normal-std1'
        elif id(p)in down:
            bound=math.sqrt(3/down[id(p)])  # kaiming_uniform_(a=1), gain1: variance1/fan_in.
            distribution,role=pg_init.Uniform(-bound,bound),'original-a1-down-QR'
        else:distribution,role=pg_init.KEEP,'protected-zeroUP-KEEP'if id(p)in up else'original-routed-bridge-router-KEEP'
        rows.append(dict(name='slider.'+name,parameter=p,distribution=distribution,role=role))
    rows.extend(dict(name='critic.'+n,parameter=p,distribution=pg_init.KEEP,role='original-Linear-AudioResponseCritic-constructor-KEEP')for n,p in critic.named_parameters(remove_duplicate=False))
    seen=set();storages=set()
    for row in rows:
        p=row['parameter'];storage=(str(p.device),p.untyped_storage().data_ptr())
        if(type(p)is not nn.Parameter or not p.requires_grad or p.grad is not None or p.dtype!=torch.float32
                or not p.is_contiguous()or not torch.isfinite(p).all()or id(p)in seen
                or p.numel()and storage in storages or hasattr(p,MARKER)):raise ValueError('unique materialized fresh FP32 owned Parameters required')
        seen.add(id(p));storages.add(storage)
    s={id(p)for p in slider.parameters()};c={id(p)for p in critic.parameters()}
    if seen!=s|c or s&c or not set(down)<=seen or not set(up)<=seen:raise ValueError('all and only added slider/critic Parameters declared once')
    return rows

def initialize(slider,critic,construction,options,sources):
    if(not isinstance(construction,Construction)or construction.mode!='fresh'or not construction.cohort
            or construction.options_sha256!=hashlib.sha256(canonical(options).encode()).hexdigest()):raise ValueError('explicit fresh intent before optimizers required')
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:raise ValueError('exact public PG095 initializer required')
    rows=declarations(slider,critic);ids=[id(r['parameter'])for r in rows];before=[tensor_sha(r['parameter'])for r in rows]
    holder=DeclaredOwned(rows);seeds={'p'+str(i):seed_for(options['config']['seed'],r['name'])for i,r in enumerate(rows)}
    pg_init.deterministic_orthogonal_(holder,strict=True,parameter_seeds=seeds)
    if ids!=[id(r['parameter'])for r in rows]:raise RuntimeError('original added Parameter identity changed')
    parameters=[]
    for i,row in enumerate(rows):
        after=tensor_sha(row['parameter'])
        if row['distribution']is pg_init.KEEP and before[i]!=after:raise RuntimeError('deliberate original constructor law changed')
        parameters.append(dict(name=row['name'],role=row['role'],shape=list(row['parameter'].shape),dtype=str(row['parameter'].dtype),distribution=repr(row['distribution']),seed=seeds['p'+str(i)],before_sha256=before[i],after_sha256=after))
    receipt=dict(schema=SCHEMA,profile=PROFILE,lane=options['config']['lane'],cohort=construction.cohort,options_sha256=construction.options_sha256,
        source_identity=sources,seed=options['config']['seed'],completed_steps=0,restored_from=None,initializer=source_record(pg_init.__file__),parameters=parameters,
        parameter_identity_preserved=True,host_down_kaiming_a=1,host_down_target_mean_square='1/fan_in',bank='same-original-128x4-R2Normal(0,1)',
        other_added_and_critic='original-constructor-KEEP',selection='raw',ema_decay=None)
    marker=hashlib.sha256(canonical(receipt).encode()).hexdigest()
    for r in rows:setattr(r['parameter'],MARKER,marker)
    return receipt

def validate_receipt(receipt,construction,options,sources):
    if(not isinstance(receipt,dict)or receipt.get('schema')!=SCHEMA or receipt.get('profile')!=PROFILE
            or receipt.get('lane')!=options['config']['lane']or receipt.get('cohort')!=construction.cohort
            or receipt.get('options_sha256')!=construction.options_sha256 or receipt.get('source_identity')!=sources
            or receipt.get('seed')!=options['config']['seed']or receipt.get('completed_steps')!=0 or receipt.get('restored_from')is not None
            or receipt.get('selection')!='raw'or receipt.get('ema_decay')is not None or receipt.get('initializer',{}).get('sha256')!=INIT_SHA):raise ValueError('exact new-cohort initialization receipt required')
