"""Owner-authenticated completed original AR responses only; never collect teachers."""
from dataclasses import fields
from pathlib import Path
import torch
from particle_sliders.pg09_audio import AudioRow,PreparedAudioResponse,audio_conditions
from . import PROFILE,ownership
from .util import canonical,equal,read_packet,save_packet,source_record,tree
SCHEMA='YuE2-completed-original-PreparedAudioResponses-fresh95-cache-v1'

def file_record(row):
    if(not isinstance(row,dict)or set(row)!={'path','bytes','sha256'}or not isinstance(row['path'],str)
        or not Path(row['path']).is_absolute()or'..'in Path(row['path']).parts or type(row['bytes'])is not int or row['bytes']<1
        or not isinstance(row['sha256'],str)or len(row['sha256'])!=64 or any(x not in'0123456789abcdef'for x in row['sha256'])):raise ValueError('whole original file path/bytes/SHA required')

def teacher_identity(config):return {k:config[k]for k in('lane','model_identity','width','max_sequence_length')}
def validate_prepared(prepared,config):
    if not isinstance(prepared,(tuple,list))or not 2<=len(prepared)<=512:raise ValueError('2..512 completed original PreparedAudioResponses required')
    maximum=config['max_sequence_length'];vocab=256 if config['lane']=='cpu-fixture'else 184704
    for item in prepared:
        if type(item)is not PreparedAudioResponse or type(item.row)is not AudioRow:raise ValueError('original typed PreparedAudioResponse/AudioRow only')
        row=item.row
        if type(row.neutral_ids)is not tuple or type(row.positive_ids)is not tuple or row.scale!=1.:raise ValueError('complete original tuples/unit-positive target required')
        for ids in(row.neutral_ids,row.positive_ids):
            if not 1<=len(ids)<=maximum or any(type(x)is not int or not 0<=x<vocab for x in ids):raise ValueError('whole original unpadded token IDs; no truncation')
        if(row.neutral_ids==row.positive_ids or type(row.history_tokens)is not int or not 0<=row.history_tokens<min(len(row.neutral_ids),len(row.positive_ids))
            or row.history_tokens and row.neutral_ids[-row.history_tokens:]!=row.positive_ids[-row.history_tokens:]):raise ValueError('distinct prompts with exact declared shared causal continuation required')
        for value in(item.target,item.neutral):
            if(type(value)is not torch.Tensor or value.shape!=(1,config['width'])or value.device.type!='cpu' or value.dtype!=torch.float32
                or value.requires_grad or value.layout!=torch.strided or not torch.isfinite(value).all()):raise ValueError('complete frozen CPU FP32 final-AR fields only; not audio PCM')

def dump(item):return dict(row={f.name:getattr(item.row,f.name)for f in fields(AudioRow)},target=tree(item.target,'cpu'),neutral=tree(item.neutral,'cpu'))
def restore(row,device):
    if not isinstance(row,dict)or set(row)!={'row','target','neutral'}or not isinstance(row['row'],dict)or set(row['row'])!={f.name for f in fields(AudioRow)}:raise ValueError('complete serialized original AudioRow required')
    return PreparedAudioResponse(AudioRow(**row['row']),tree(row['target'],device),tree(row['neutral'],device))
def to_device(prepared,device):return[restore(dump(item),device)for item in prepared]

def origins_valid(origins,prepared,config,scope):
    if not isinstance(origins,list)or len(origins)!=len(prepared):raise ValueError('each completed target needs original origins')
    for row in origins:
        if(not isinstance(row,dict)or row.get('scope')!=scope or row.get('complete_prepared_output')is not True
            or row.get('teacher_identity')!=teacher_identity(config)or not isinstance(row.get('whole_output_files'),list)or not row['whole_output_files']
            or not isinstance(row.get('source_files'),list)or not row['source_files']):raise ValueError('authenticated complete native/source/token-context origins required')
        for record in row['whole_output_files']+row['source_files']:file_record(record)
    canonical(origins)

def save_prepared_cache(path,*,config,prepared,teacher_origins,cache_id,scope,owner_admission):
    from .api import configuration
    config=configuration(config);ownership.guard();validate_prepared(prepared,config)
    if scope not in('completed-native','synthetic-software-only')or config['lane']=='native2048'and scope!='completed-native':raise ValueError('synthetic fixture is not native availability/qualification')
    origins_valid(teacher_origins,prepared,config,scope)
    if not isinstance(cache_id,str)or not cache_id.strip()or not callable(owner_admission):raise ValueError('cache ID and authenticating owner required')
    request=dict(operation='serialize-completed-cache',profile=PROFILE,cache_id=cache_id,scope=scope,teacher_identity=teacher_identity(config),teacher_origins=teacher_origins,model_or_teacher_calls=0)
    if owner_admission(request)is not True:raise ValueError('owner must authenticate actual completed outputs and original token/packing/base provenance')
    return save_packet(path,dict(schema=SCHEMA,profile=PROFILE,cache_id=cache_id,scope=scope,teacher_identity=teacher_identity(config),teacher_origins=teacher_origins,
        original_sources=ownership.sources()['original'],serializer_source=source_record(__file__),rows=[dump(x)for x in prepared],conditions=audio_conditions(prepared).detach().cpu(),complete=True,model_or_teacher_calls=0))

def load_cache(pin,*,config):
    packet=read_packet(pin)
    keys={'schema','profile','cache_id','scope','teacher_identity','teacher_origins','original_sources','serializer_source','rows','conditions','complete','model_or_teacher_calls'}
    if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=SCHEMA or packet['profile']!=PROFILE
        or packet['complete']is not True or packet['model_or_teacher_calls']!=0 or packet['scope']not in('completed-native','synthetic-software-only')
        or config['lane']=='native2048'and packet['scope']!='completed-native' or packet['teacher_identity']!=teacher_identity(config)
        or packet['original_sources']!=ownership.sources()['original']or packet['serializer_source']!=source_record(__file__)):raise ValueError('exact completed cache/native/lane/Source required; no old-cohort migration')
    prepared=[restore(x,'cpu')for x in packet['rows']];validate_prepared(prepared,config);origins_valid(packet['teacher_origins'],prepared,config,packet['scope'])
    if not equal(audio_conditions(prepared),packet['conditions']):raise ValueError('original audio_conditions mismatch')
    identity={k:packet[k]for k in('schema','profile','cache_id','scope','teacher_identity','teacher_origins','original_sources','serializer_source')}
    identity.update(whole_file=dict(path=pin.path,bytes=pin.bytes,sha256=pin.sha256),row_count=len(prepared),response_shape=[1,config['width']],game_response_scale=1.0)
    return prepared,packet['conditions'],identity
