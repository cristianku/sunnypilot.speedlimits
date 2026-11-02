"""Real resolver and SLA classes, platform imports isolated; no daemon/device execution.

Compile the source AST without imports because this server lacks the runtime's
native msgq/params modules. Schema enums and the planner's validity expression
come from the pinned source, not from a reimplementation of the algorithms.
"""
import ast
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
import time

import pytest

def execute(path, env):
    tree = ast.parse(Path(path).read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), 'exec'), env)
    return env


@pytest.fixture
def chain(candidate):
    import capnp
    from importlib.util import spec_from_file_location, module_from_spec
    runtime = Path('/root/sunnypilot')
    # One schema load per candidate path, avoiding duplicate Cap'n Proto IDs.
    if not hasattr(chain, 'custom'):
        chain.custom = capnp.load(str(candidate/'openpilot/cereal/custom.capnp'),
                                  imports=[str(runtime/'openpilot/cereal')])
    custom = chain.custom
    class IntEnumBase(IntEnum):
        @classmethod
        def min(cls): return min(cls)
        @classmethod
        def max(cls): return max(cls)
    common = execute(runtime/'openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/common.py',
                     {'IntEnumBase':IntEnumBase})
    constants = execute(runtime/'openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/__init__.py', {})
    cv_spec=spec_from_file_location('units_cv',runtime/'openpilot/common/constants.py')
    cv=module_from_spec(cv_spec);cv_spec.loader.exec_module(cv)
    model_spec=spec_from_file_location('units_model_constants',runtime/'openpilot/selfdrive/modeld/constants.py')
    model=module_from_spec(model_spec);model_spec.loader.exec_module(model)
    class Params:
        values = {'SpeedLimitPolicy':6,'SpeedLimitOffsetType':0,'SpeedLimitValueOffset':0,
                  'SpeedLimitMode':3, 'IsMetric':True}
        def get(self, name, **kw): return self.values.get(name,0)
        def get_bool(self, name): return bool(self.values.get(name,False))
    env = dict(common, **constants, custom=custom, time=time, CV=cv.CV, Params=Params,
               PARAMS_UPDATE_PERIOD=3, DT_MDL=.05, messaging=NS(SubMaster=object),
               get_gps_location_service=lambda _: 'gpsLocationExternal',
               get_sanitize_int_param=lambda k,low,high,p: p.get(k),
               car=NS(CarParams=object,CarState=NS(ButtonEvent=NS(Type=NS(accelCruise=1,resumeCruise=2,
                       decelCruise=3,setCruise=4)))), CONTROL_N=17, ModelConstants=model.ModelConstants,
               EventsSP=object, Mode=common['Mode'])
    # Helper logic is real too; only its Params storage is substituted.
    helper=execute(runtime/'openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/helpers.py',
                   dict(env, SpeedLimitMode=common['Mode']))
    env.update({k:helper[k] for k in ('compare_cluster_target','set_speed_limit_assist_availability')})
    prefix=candidate/'openpilot/sunnypilot/selfdrive/controls/lib/speed_limit'
    resolver_env=execute(prefix/'speed_limit_resolver.py',dict(env))
    sla_env=execute(prefix/'speed_limit_assist.py',dict(env))
    resolver=resolver_env['SpeedLimitResolver']()
    cp=NS(openpilotLongitudinalControl=True,pcmCruise=True,brand='peugeot')
    sla=sla_env['SpeedLimitAssist'](cp,NS(pcmCruiseSpeed=False))
    data={'carStateSP':NS(speedLimit=99), 'speedVisionSP':NS(speedLimit=0),
          'liveMapDataSP':NS(speedLimit=0,speedLimitValid=False,speedLimitAhead=0,
                          speedLimitAheadValid=False,speedLimitAheadDistance=0)}
    class SM:
        alive={'speedVisionSP':True}
        recv_time={'gpsLocationExternal':time.monotonic(),'liveMapDataSP':time.monotonic()}
        def __getitem__(self,k):return data[k]
    return resolver,sla,SM(),data,Params,custom


def step(chain, speed, map_limit=0):
    r,s,sm,data,params,custom=chain
    data['speedVisionSP'].speedLimit=speed
    data['liveMapDataSP'].speedLimit=map_limit
    data['liveMapDataSP'].speedLimitValid=map_limit>0
    for _ in range(65):  # cross the real 3-second parameter refresh interval
        r.update(20,sm)
        s.update(True,False,20,0,40,r.speed_limit,r.speed_limit_final_last,
                 r.speed_limit_valid or r.speed_limit_last_valid,r.distance,NS(add=lambda _:None))
    return s.output_v_target


@pytest.mark.parametrize('policy',[5,6])
def test_zero_revokes_previous_target_through_resolver_and_sla(chain,policy):
    r,s,sm,data,params,_=chain
    params.values['SpeedLimitPolicy']=policy
    r.policy=policy
    assert step(chain,22.352) == pytest.approx(22.352)
    assert r.speed_limit_last_valid
    assert step(chain,0) == 255
    assert (r.speed_limit,r.speed_limit_last,r.speed_limit_final_last)==(0,0,0)
    assert not r.speed_limit_last_valid
    assert not s._has_speed_limit
    assert step(chain,11.176) == pytest.approx(11.176)


@pytest.mark.parametrize('policy',[5,6])
def test_map_fallback_and_display_units(chain,policy):
    r,s,sm,data,params,_=chain
    params.values['SpeedLimitPolicy']=policy
    r.policy=policy
    assert step(chain,13.4112,30) == pytest.approx(13.4112)
    params.values['IsMetric']=False
    assert step(chain,13.4112,30) == pytest.approx(13.4112)
    assert r.speed_limit==pytest.approx(13.4112) and s.speed_limit_final_last_conv==30
    target=step(chain,0,30)
    assert target == (30 if policy==5 else 255)
    assert r.source == (2 if policy==5 else 0)  # schema's map / none


def test_stale_vision_does_not_retain_target(chain):
    assert step(chain,22.352)==pytest.approx(22.352)
    chain[2].alive['speedVisionSP']=False
    assert step(chain,22.352)==255


@pytest.mark.parametrize('policy',[5,6])
def test_memory_country_end_and_sla_chain(chain,vision,policy):
    r,s,sm,data,params,_=chain
    params.values['SpeedLimitPolicy']=policy
    r.policy=policy
    m=vision.SpeedLimitMemory('US')
    m.update(0,[25]);m.update(.2,[25])
    assert step(chain,m.speed_limit,30)==pytest.approx(11.176)
    m.set_country(None)
    m.update(1,[50]);m.update(1.2,[50])
    assert step(chain,m.speed_limit,30)==pytest.approx(11.176)
    m.update(2,[0]);m.update(2.2,[0])
    assert step(chain,m.speed_limit,30)==(30 if policy==5 else 255)
    m.set_country('GB')
    m.update(3,[50]);m.update(3.2,[50])
    assert step(chain,m.speed_limit)==pytest.approx(22.352)
    m.set_country('FR')
    assert step(chain,m.speed_limit)==255


def test_schema_roundtrip_old_and_new_logs(chain,vision,tmp_path):
    import capnp
    custom=chain[-1]
    fields=custom.SpeedVisionSP.schema.fields
    old_fields={'speedLimit':0,'modelName':1,'signRead':2,'roadType':3,'speedLimitState':4,
                'speedLimitReason':5,'rawSignClass':6,'rawSignSpeed':7,'rawSignConfidence':8,
                'rawSignTaken':9,'signDistance':10,'signSize':11}
    assert all(fields[n].proto.ordinal.explicit==i for n,i in old_fields.items())
    old=tmp_path/'old.capnp'
    old.write_text('''@0xbfc25c122684ff27;
struct OldSpeedVisionSP {
speedLimit @0 :Float32; modelName @1 :Text; signRead @2 :Bool; roadType @3 :Text;
speedLimitState @4 :Text; speedLimitReason @5 :Text; rawSignClass @6 :Text;
rawSignSpeed @7 :Float32; rawSignConfidence @8 :Float32; rawSignTaken @9 :Bool;
signDistance @10 :Float32; signSize @11 :Float32;
}''')
    previous=capnp.load(str(old)).OldSpeedVisionSP
    before=previous.new_message(speedLimit=22.2222222,rawSignSpeed=80)
    with custom.SpeedVisionSP.from_bytes(before.to_bytes()) as m:
        assert m.rawSignSpeed==80 and m.rawSignValue==0 and m.rawSignUnit==m.rawSignCountry==''
    memory=vision.SpeedLimitMemory('US')
    raw={'class':'maximum_speed:25','confidence':.99}
    memory.update(0,[25],raw=raw);memory.update(.2,[25],raw=raw)
    m=custom.SpeedVisionSP.new_message(speedLimit=memory.speed_limit)
    memory.fill_log(m)
    payload=m.to_bytes()
    with previous.from_bytes(payload) as legacy:
        assert legacy.speedLimit==pytest.approx(11.176) and legacy.rawSignSpeed==pytest.approx(40.2336)
    with custom.SpeedVisionSP.from_bytes(payload) as current:
        assert current.rawSignValue==25 and current.rawSignUnit=='mph' and current.rawSignCountry=='US'
