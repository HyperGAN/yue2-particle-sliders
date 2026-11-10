"""Installed original guards and exact constructor-owned reversible YuE2 state."""
from importlib import metadata
from pathlib import Path
import base64,csv,io
from .baseline import ORIGINAL
from .util import source_record

MARKER='_yue2_explicit_cached_fresh95_owner'
ATTRS=('forward','_pg09_scale_checkpoint','gradient_checkpointing','_gradient_checkpointing_func')

def distribution(name,version):
    normal=lambda s:s.lower().replace('_','-')
    rows=[d for d in metadata.distributions()if normal(d.metadata.get('Name')or'')==normal(name)]
    if len(rows)!=1 or rows[0].version!=version:raise RuntimeError('one exact owning distribution required:'+name)
    return rows[0]

def record_guard(dist,namespace,root,pins):
    if Path(dist.locate_file(namespace+'/__init__.py')).absolute()!=root/'__init__.py':raise RuntimeError('genuine installed namespace required:'+namespace)
    recorded={r[0]:r[1:]for r in csv.reader(io.StringIO(dist.read_text('RECORD')or''))}
    for name,row in pins.items():
        actual=source_record(root/name)
        if actual!=row:raise RuntimeError('exact captured Source differs:'+name)
        digest=base64.urlsafe_b64encode(bytes.fromhex(actual['sha256'])).rstrip(b'=').decode()
        if recorded.get(namespace+'/'+name)!=['sha256='+digest,str(actual['bytes'])]:raise RuntimeError('genuine owning RECORD required:'+name)

def guard():
    import yue2_sliders
    from yue2_sliders.response_runtime import verify_runtime
    from particlegan import init as pg_init
    from .policy import INIT_SHA
    record_guard(distribution('yue2-particle-sliders','0.4.0'),'yue2_sliders',Path(yue2_sliders.__file__).absolute().parent,ORIGINAL)
    lock=verify_runtime()  # Original canonical577 + actual distribution/import-origin guards unchanged.
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:raise RuntimeError('exact public PG095 initializer required')
    own=Path(__file__).absolute().parent
    record_guard(distribution('yue2-explicit-fresh95','0.1.0'),'yue2_fresh95',own,
        {p.name:source_record(p)for p in sorted(own.glob('*.py'))})
    return dict(core=lock,PG='0.9.5',original_guard_unchanged=True,genuine_original_and_addon_RECORD=True)

def sources():
    import yue2_sliders
    from particle_sliders import reference,pg09_audio,pg09,pg095,pg09_backends,lifecycle
    root=Path(yue2_sliders.__file__).absolute().parent
    return dict(own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))},original={n:source_record(root/n)for n in ORIGINAL},
        core={m.__name__:source_record(m.__file__)for m in(reference,pg09_audio,pg09,pg095,pg09_backends,lifecycle)})

class Attachment:
    def __init__(self,runtime):
        self.runtime=runtime;model=runtime.model
        if hasattr(model,MARKER):raise ValueError('another YuE2 owner exists')
        if any('forward'in vars(m)or m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('pristine original bare host forwards required')
        self.model=model;self.slider=self.adapter=None;self.closed=self.closing=False
        self.host_parameters=tuple(model.named_parameters());self.modules=tuple(model.modules());self.flags=[(p,p.requires_grad)for p in model.parameters()];self.modes=[(m,m.training)for m in self.modules]
        self.before=[(m,k,k in vars(m),vars(m).get(k))for m in self.modules for k in ATTRS]
        self.pending_attrs=[];self.containers=[];self.pending_hooks=[];self.pending_flags=list(self.flags);self.pending_modes=list(self.modes);self.pending_marker=True
        self.owned_flags={id(p):f for p,f in self.flags};self.owned_modes={id(m):f for m,f in self.modes}
        setattr(model,MARKER,self)

    def construct(self,config):
        from yue2_sliders.slider import ParticleSlider
        from yue2_sliders.stamp import locked_stamp
        from yue2_sliders.response_native import NativeResponse,native_initialization_rng
        from .api import original_options
        try:
            with native_initialization_rng(original_options(config)):
                self.model.eval().requires_grad_(False)
                self.slider=ParticleSlider(self.model,locked_stamp()[0])
                self.adapter=NativeResponse(self.runtime,self.slider,model_identity=config['model_identity'],
                    checkpoint_modules=tuple(self.model.model.layers)if config['checkpointing']and config['lane']=='native2048'else(),
                    pg_source_profile=config['pg_source_profile'],penalty_input_domain=config['penalty_input_domain'])
        finally:self._capture_constructor_state()
        self.ids=[(n,id(p))for n,p in self.slider.named_parameters()]
        self.loras=tuple(self.slider.adapters.values());self.originals=[(l,l.org_forward)for l in self.loras]
        return self.adapter

    def _capture_constructor_state(self):
        self.pending_attrs=[(m,k,had,old,k in vars(m),vars(m).get(k))for m,k,had,old in self.before if(k in vars(m))!=had or vars(m).get(k)is not old]
        self.owned_flags={id(p):p.requires_grad for p,_ in self.flags};self.owned_modes={id(m):m.training for m,_ in self.modes}
        handles=getattr(self.adapter,'_checkpointing',None)
        if handles is not None:
            if type(handles)is not list:raise RuntimeError('original private checkpoint list required')
            self.containers.append((self.adapter,'_checkpointing',handles,tuple(handles)))

    def _containers_unchanged(self):
        for owner,key,container,items in self.containers:
            current=getattr(owner,key,None)
            if current is not container or len(current)!=len(items)or any(a is not b for a,b in zip(current,items)):raise RuntimeError('foreign private handle container/items retained')

    def _originals_unchanged(self):
        current=tuple(self.model.named_parameters())
        if len(current)!=len(self.host_parameters)or any(n!=oldn or p is not oldp for (n,p),(oldn,oldp)in zip(current,self.host_parameters)):raise RuntimeError('foreign original model Parameter retained')
        if hasattr(self,'loras')and(tuple(self.slider.adapters.values())!=self.loras or any(l.org_forward is not f for l,f in self.originals)):raise RuntimeError('foreign original LoRA callback/list retained')

    def require_active(self):
        if self.closed or self.closing:raise RuntimeError('YuE2 closed/cleanup pending')
        if vars(self.model).get(MARKER)is not self:raise RuntimeError('foreign YuE2 marker')
        self._originals_unchanged()
        if [(n,id(p))for n,p in self.slider.named_parameters()]!=self.ids:raise RuntimeError('original adapter/bank Parameter identity changed')
        if tuple(self.slider.adapters.values())!=self.loras or any(l.org_forward is not f for l,f in self.originals):raise RuntimeError('foreign original LoRA callback/list')
        if any(p.requires_grad is not self.owned_flags[id(p)]for p,_ in self.flags)or any(m.training is not self.owned_modes[id(m)]for m,_ in self.modes):raise RuntimeError('foreign host gradient flag/mode')
        self._containers_unchanged()
        for m,k,_,_,had,owned in self.pending_attrs:
            if(k in vars(m))!=had or vars(m).get(k)is not owned:raise RuntimeError('foreign native/checkpoint attribute:'+k)
        for handle,hooks,callback,extras in self.pending_hooks:
            if handle.hooks_dict_ref()is not hooks or hooks.get(handle.id)is not callback or any((handle.id in x)!=had or x.get(handle.id)is not owned for x,had,owned in extras):raise RuntimeError('foreign private callback')

    @staticmethod
    def _run(pending,operation,errors):
        failed=[]
        for value in pending:
            try:operation(value)
            except BaseException as error:failed.append(value);errors.append(error)
        return failed

    def close(self):
        if self.closed:return
        self.closing=True
        if not self.pending_marker or vars(self.model).get(MARKER)is not self:raise RuntimeError('foreign YuE2 marker retained before any borrowed cleanup')
        self._originals_unchanged()
        errors=[]
        try:self._containers_unchanged()
        except BaseException as error:errors.append(error)
        else:
            def hook(row):
                handle,hooks,callback,extras=row
                if handle.hooks_dict_ref()is not hooks or hooks.get(handle.id)is not callback or any((handle.id in x)!=had or x.get(handle.id)is not owned for x,had,owned in extras):raise RuntimeError('foreign private callback retained')
                handle.remove()
            self.pending_hooks=self._run(self.pending_hooks,hook,errors)
            def attribute(row):
                m,k,had,old,has_owned,owned=row
                if(k in vars(m))!=has_owned or vars(m).get(k)is not owned:raise RuntimeError('foreign native/checkpoint attribute retained:'+k)
                if had:setattr(m,k,old)
                else:vars(m).pop(k,None)
            self.pending_attrs=self._run(self.pending_attrs,attribute,errors)
        if not self.pending_hooks and not self.pending_attrs and not errors:
            self._containers_unchanged()
            for _,_,container,_ in self.containers:container.clear()
            self.containers=[]
            def flag(row):
                p,old=row
                if p.requires_grad is not self.owned_flags[id(p)]:raise RuntimeError('foreign host gradient flag retained')
                p.requires_grad_(old)
            def mode(row):
                m,old=row
                if m.training is not self.owned_modes[id(m)]:raise RuntimeError('foreign host mode retained')
                m.training=old
            self.pending_flags=self._run(self.pending_flags,flag,errors);self.pending_modes=self._run(self.pending_modes,mode,errors)
            if not self.pending_flags and not self.pending_modes:
                if vars(self.model).get(MARKER)is not self:errors.append(RuntimeError('foreign marker retained'))
                else:delattr(self.model,MARKER);self.pending_marker=False
        if errors:raise BaseExceptionGroup('YuE2 cleanup pending',errors)
        if any((self.pending_hooks,self.pending_attrs,self.pending_flags,self.pending_modes,self.containers))or self.pending_marker:raise RuntimeError('YuE2 cleanup pending')
        self.closed,self.closing=True,False

class Owner:
    def __init__(self):self.closed=self.closing=False;self.attachment=None
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('YuE2 owner closed/cleanup pending')
        if self.attachment is not None:self.attachment.require_active()
    def close(self):
        if self.closed:return
        self.closing=True
        if self.attachment is not None:self.attachment.close()
        self.closed,self.closing=True,False
    def __enter__(self):self.require_open();return self
    def __exit__(self,kind,primary,traceback):
        try:self.close()
        except BaseException as cleanup:
            if primary is not None:raise BaseExceptionGroup('YuE2 work and cleanup failed',[primary,cleanup])
            raise
