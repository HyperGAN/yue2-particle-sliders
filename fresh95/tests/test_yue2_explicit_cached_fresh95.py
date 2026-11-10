"""FOUR NEW UNRUN CPU software nodes/ONE update; no native audio qualification."""
from copy import deepcopy
import math
import pytest,torch
from yue2_sliders.backend import YuE2Backend
from particle_sliders.pg09_audio import AudioRow,PreparedAudioResponse,YuE2ARResponseAdapter
from particle_sliders.lifecycle import UnsupportedOperation
from yue2_sliders import lifecycle as original
from yue2_fresh95 import open_training,open_inference,save_prepared_cache,file_pin,Checkpoint
from yue2_fresh95 import api,cache,policy,ownership
from yue2_fresh95.util import equal

def bare(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return YuE2Backend(dummy=True,device='cpu')

def factory(request):
    assert request['operation']in('fresh','resume','inference')and request['new_teacher_or_tokenizer_calls']==0
    assert request['options']['config']['lane']=='cpu-fixture'
    return bare(request['options']['config']['seed'])

@pytest.fixture
def staged(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('no teacher/old factory/tokenizer/native/acoustic path')
    monkeypatch.setattr(original,'prepare',forbidden);monkeypatch.setattr(original,'_prepare',forbidden);monkeypatch.setattr(original,'build_native',forbidden)
    monkeypatch.setattr(YuE2ARResponseAdapter,'prepare',forbidden)
    runtime=bare(7);base=deepcopy(runtime.model.state_dict())
    config=dict(lane='cpu-fixture',model_identity={},steps=3,batch=2,seed=7,width=32,device='cpu',base_precision='float32',checkpointing=False,
        max_sequence_length=256,pg_source_profile='release-0.9.5',penalty_input_domain='auto',preset='bispf_optimized',ema_decay=None,lr=.001,d_lr_mult=18.,prior_lr_mult=30.)
    config['model_identity']=api.model_identity(runtime,config)
    prepared=[PreparedAudioResponse(AudioRow((11+i,17,19),(29+i,17,19),2,1.),torch.linspace(-.15,.3,32).reshape(1,32)+i*.01,torch.linspace(-.2,.1,32).reshape(1,32))for i in range(3)]
    data=tmp_path/'NEW-completed-synthetic-fields.pt';torch.save([cache.dump(x)for x in prepared],data)
    whole=file_pin(data);source=file_pin(__file__)
    origins=[dict(scope='synthetic-software-only',complete_prepared_output=True,synthetic_row=i,teacher_identity=cache.teacher_identity(config),
        whole_output_files=[dict(path=whole.path,bytes=whole.bytes,sha256=whole.sha256)],source_files=[dict(path=source.path,bytes=source.bytes,sha256=source.sha256)],real_teacher_calls=0)for i in range(3)]
    retained=save_prepared_cache(tmp_path/'NEW-completed-cache.pt',config=config,prepared=prepared,teacher_origins=origins,cache_id='NEW-YuE2-software-completed-fields',scope='synthetic-software-only',owner_admission=lambda request:True)
    return dict(config=config,cache=retained,cohort_id='NEW-YuE2-software-cohort',component_factory=factory,owner_admission=lambda request:True),tmp_path,base

def opened(staged,name):
    options,root,_=staged
    return open_training(directory=root/name,mode='fresh',**options)

def test_new_same_Parameter_a1_down_R2bank_zeroUP_critic_router_KEEP_before_optimizers(staged,monkeypatch):
    initialize=policy.initialize;game=api.ConditionalPG095;seen=[];_,root,_=staged
    def initializing(slider,critic,*args):
        ids=[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()];before=deepcopy(critic.state_dict())
        receipt=initialize(slider,critic,*args)
        assert ids==[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()]and equal(before,critic.state_dict())
        assert all(torch.count_nonzero(l.lora_up.weight)==0 for l in slider.adapters.values())
        assert all(r['before_sha256']==r['after_sha256']for r in receipt['parameters']if r['distribution']=='KEEP')
        assert any(r['role']=='same-original-bank-R2Normal-std1'for r in receipt['parameters'])
        seen.append(receipt);return receipt
    def constructing(backend,*args,**kwargs):
        assert seen and backend.metadata()['fresh_initialization']['receipt']==seen[-1]
        assert (root/'before-opts'/'initialization.json').is_file()
        assert kwargs['particles']is backend.backend.slider.particles is backend.network.particles
        assert kwargs['response_scale']==1.0
        return game(backend,*args,**kwargs)
    monkeypatch.setattr(policy,'initialize',initializing);monkeypatch.setattr(api,'ConditionalPG095',constructing)
    with opened(staged,'before-opts')as owner:
        assert owner.game.completed_steps==0 and owner.game.prior.z is owner.slider.particles
        assert owner.game.state_dict()['config']['backend']['fresh_initialization']['receipt']==seen[-1]
        assert {id(p)for p in owner.slider.parameters()}=={id(p)for p in owner.network.parameters()}
        assert (owner.directory/'FULL-000000.pt').is_file()and(owner.directory/'RAW-000000.pt').is_file()

def test_new_native_a1_variance_Gram_R2_determinism_caller_RNG_and_base(staged):
    before=torch.get_rng_state().clone();_,_,base=staged
    with opened(staged,'laws')as owner:
        first=deepcopy(owner.network.state_dict());model=owner.model;attachment=owner.attachment
        expected=torch.special.ndtri(policy.pg_init._qr.r2_points(128,4)).to(torch.float32)
        assert torch.equal(owner.slider.particles.detach().cpu(),expected)
        for lora in owner.slider.adapters.values():
            down=lora.lora_down.weight.detach();rows,cols=down.shape;variance=1/cols
            gram=down@down.T if rows<=cols else down.T@down
            assert torch.allclose(gram,torch.eye(min(rows,cols))*variance*max(rows,cols),rtol=1e-5,atol=1e-6)
            assert math.isclose(float(down.square().mean()),variance,rel_tol=1e-5,abs_tol=1e-7)
            assert torch.count_nonzero(lora.lora_up.weight)==0
        assert equal(base,model.state_dict())
    assert torch.equal(before,torch.get_rng_state())and not hasattr(model,ownership.MARKER)
    assert all(p.requires_grad is flag for p,flag in attachment.flags)and all(m.training is mode for m,mode in attachment.modes)
    assert not any('forward'in vars(m)for m in model.modules())
    with opened(staged,'again')as again:assert equal(first,again.network.state_dict())
    assert torch.equal(before,torch.get_rng_state())

def test_new_ONE_update_whole_FULL_optimizer_cursors_RNG_RAW_no_cache_or_initializer(staged,monkeypatch):
    options,root,base=staged
    with opened(staged,'one-update')as trained:
        trained.update()  # The ONE total successful tiny software update.
        full=trained.save(root/'FULL1.pt');raw=trained.export(root/'RAW1.pt');state=deepcopy(trained.game.state_dict());network=deepcopy(trained.network.state_dict());model=trained.model
    assert equal(base,model.state_dict())
    def forbidden(*args,**kwargs):raise AssertionError('initializer/cache/critic/game forbidden on this path')
    monkeypatch.setattr(policy,'initialize',forbidden)
    with open_training(directory=root/'resume',mode='resume',resume=full,**options)as resumed:
        assert resumed.inherited_steps==1 and equal(state,resumed.game.state_dict())
    monkeypatch.setattr(api,'load_cache',forbidden);monkeypatch.setattr(api,'ConditionalPG095',forbidden)
    with open_inference(raw,component_factory=factory,owner_admission=lambda request:True)as inferred:
        assert inferred.completed_steps==1 and not hasattr(inferred,'game')and not hasattr(inferred,'critic')and equal(network,inferred.network.state_dict())
        before=torch.get_rng_state().clone();value=inferred.response(token_ids=(11,17,19),strength=.5,seed=17)
        assert value.shape==(1,32)and torch.isfinite(value).all()and torch.equal(before,torch.get_rng_state())
        assert equal(base,inferred.model.state_dict())
        with pytest.raises(UnsupportedOperation):inferred.response(token_ids=(11,17,19),strength=1.1,seed=17)  # Original formal0..1 bound.

def test_new_early_intent_cache_EMA_and_foreign_safe_private_cleanup_retry_markerLAST(staged):
    options,root,_=staged;calls=[]
    def forbidden(request):calls.append(request);raise AssertionError('invalid request reached component factory')
    bad=dict(options,component_factory=forbidden)
    for mode,resume,ema in(('fresh',Checkpoint('unused',1,'0'*64),None),('resume',None,None),('fresh',None,.9)):
        with pytest.raises(ValueError):open_training(directory=root/'bad',mode=mode,resume=resume,ema_decay=ema,**bad)
    cfg=deepcopy(options['config']);cfg['ema_decay']=.9
    with pytest.raises(ValueError):open_training(directory=root/'ema',mode='fresh',**dict(bad,config=cfg))
    wrong=deepcopy(options['config']);wrong['model_identity']['frozen_base_sha256']='0'*64
    with pytest.raises(ValueError):open_training(directory=root/'wrong-cache',mode='fresh',**dict(bad,config=wrong))
    assert calls==[]
    owner=opened(staged,'cleanup');attachment=owner.attachment;model=owner.model;module=model.model.layers[0]
    callback=lambda *args:None;handle=module.register_forward_pre_hook(callback);hooks=handle.hooks_dict_ref();container=[handle]
    owner.adapter._test_owned_handles=container;attachment.containers.append((owner.adapter,'_test_owned_handles',container,(handle,)));attachment.pending_hooks.append((handle,hooks,callback,()))
    checkpoint=object();module._test_checkpoint=checkpoint;attachment.pending_attrs.append((module,'_test_checkpoint',False,None,True,checkpoint))
    foreign_callback=lambda *args:None;hooks[handle.id]=foreign_callback
    with pytest.raises(RuntimeError):owner.require_open()
    hooks[handle.id]=callback
    attrs=[(m,k,k in vars(m),vars(m).get(k))for m,k,*_ in attachment.pending_attrs];flags=[(p,p.requires_grad)for p,_ in attachment.flags];modes=[(m,m.training)for m,_ in attachment.modes]
    foreign=object();setattr(model,ownership.MARKER,foreign)
    with pytest.raises(RuntimeError):owner.close()
    assert vars(model)[ownership.MARKER]is foreign and hooks[handle.id]is callback and container==[handle]
    assert all((k in vars(m))==had and vars(m).get(k)is value for m,k,had,value in attrs)
    assert all(p.requires_grad is flag for p,flag in flags)and all(m.training is mode for m,mode in modes)
    setattr(model,ownership.MARKER,attachment);hooks[handle.id]=foreign_callback
    with pytest.raises(BaseExceptionGroup):owner.close()
    assert hooks[handle.id]is foreign_callback and vars(model)[ownership.MARKER]is attachment
    hooks[handle.id]=callback;parameter=attachment.flags[0][0];parameter.requires_grad_(True);module.training=True
    with pytest.raises(BaseExceptionGroup):owner.close()
    assert parameter.requires_grad is True and module.training is True and vars(model)[ownership.MARKER]is attachment
    parameter.requires_grad_(attachment.owned_flags[id(parameter)]);module.training=attachment.owned_modes[id(module)];owner.close();owner.close()
    assert owner.closed and not hasattr(model,ownership.MARKER)and not hooks and not container
    assert all(p.requires_grad is flag for p,flag in attachment.flags)and all(m.training is mode for m,mode in attachment.modes)
    with pytest.raises(RuntimeError):owner.update()  # Deliberate closed-owner refusal, no second update.
