"""Run speedvisiond.main with simulated camera/network/socket I/O; no inference."""
from types import SimpleNamespace as NS

import pytest
import numpy as np

from test_chain import execute


def test_daemon_publishes_memory_mps_once(candidate,vision,country):
    class Stop(Exception): pass
    clock=NS(now=0.)
    outputs=[]
    class Client:
        width=16
        height=16
        def __init__(self,*args):pass
        def connect(self,*args):return True
        def recv(self):
            clock.now+=.2
            if clock.now>1.3: raise Stop
            return NS(data=bytes(16*16*3//2),width=16,height=16,stride=16,uv_offset=256)
    def drain(sock):
        if sock=='gpsLocationExternal':
            return [NS(logMonoTime=int((clock.now-.01)*1e9),valid=True,gpsLocationExternal=NS(
                latitude=0,longitude=.5,horizontalAccuracy=5,hasFix=True))]
        return []
    class Pub:
        def __init__(self,topics):pass
        def send(self,topic,msg):outputs.append(msg.speedVisionSP)
    class Sub:
        def __init__(self,*args):pass
        def update(self,*args):pass
    settings={'country_policy':dict(gps_max_age_s=2,gps_max_accuracy_m=20,border_margin_m=100,
                                   confirm_s=.1,max_sample_gap_s=1)}
    files=NS(settings=settings,name='synthetic',finder='finder',classifier='classifier',road_type=None,
             classes=['maximum_speed:30'])
    signs=[dict(box=[0,0,10,10],confidence=.99,speed=30.,**{'class':'maximum_speed:30'})]
    v=NS(**vars(vision));v.SignNetInput=lambda *args:None
    v.read_signs=lambda *args:signs
    b=country.OfflineBorders([{'code':'GB','bbox':[0,-1,1,1],
          'polygons':[[[[0,-1],[1,-1],[1,1],[0,1],[0,-1]]]]}])
    env=dict(os=NS(nice=lambda _:None,environ={}),time=NS(monotonic=lambda:clock.now,sleep=lambda _:None),
             np=np,NoReturn=object,VisionIpcClient=Client,VisionStreamType=NS(VISION_STREAM_NARROW_ROAD=0),
             messaging=NS(sub_sock=lambda name,**kw:name,drain_sock=drain,SubMaster=Sub,PubMaster=Pub,
                          new_message=lambda *a,**kw:NS(speedVisionSP=NS())),
             Params=lambda:None,get_gps_location_service=lambda _:'gpsLocationExternal',
             cloudlog=NS(warning=lambda _:None,exception=lambda _:None),DEVICE_CAMERAS={},vision=v,
             runner=NS(networks=lambda:files,Network=lambda _:None,compile_network=lambda p:p),
             CountryResolver=country.CountryResolver,OfflineBorders=NS(load=lambda:b),GpsFix=country.GpsFix,
             __name__='units_daemon')
    execute(candidate/'openpilot/sunnypilot/speed_limits/speedvisiond.py',env)
    with pytest.raises(Stop):env['main']()
    assert outputs[-1].speedLimit==pytest.approx(13.4112)
    assert (outputs[-1].rawSignValue,outputs[-1].rawSignUnit,outputs[-1].rawSignCountry)==(30,'mph','GB')
