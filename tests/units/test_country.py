import pytest


def borders(country):
    # Synthetic adjacent polygons: FR west of lon 0, GB east. Deliberately not real geography.
    return country.OfflineBorders([
        {'code':'FR','bbox':[-1,-1,0,1], 'polygons':[[[[-1,-1],[0,-1],[0,1],[-1,1],[-1,-1]]]]},
        {'code':'GB','bbox':[0,-1,1,1], 'polygons':[[[[0,-1],[1,-1],[1,1],[0,1],[0,-1]]]]}])


def resolver(country):
    return country.CountryResolver(borders(country), country.CountryPolicy(
        gps_max_age_s=2, gps_max_accuracy_m=20, border_margin_m=100,
        confirm_s=1, max_sample_gap_s=1))


def fix(country, t, lon=-.5, **kw):
    return country.GpsFix(t, 0, lon, kw.get('accuracy', 5), kw.get('valid', True))


def test_border_jitter_stale_missing_and_confirmed_crossing(country):
    r=resolver(country)
    assert r.update(0) is None
    assert r.update(0,fix(country,0)) is None
    assert r.update(.5,fix(country,.5)) is None
    assert r.update(1,fix(country,1)) == 'FR'
    # Ambiguous or stale GPS blocks new numbers; the memory separately holds physical speed.
    assert r.update(1.2,fix(country,1.2,lon=.00001)) is None
    assert r.update(1.4,fix(country,1.4,lon=-.00001)) is None
    assert r.update(2,fix(country,2,lon=.5)) is None
    assert r.update(2.5,fix(country,2.5,lon=.5)) is None
    assert r.update(3,fix(country,3,lon=.5)) == 'GB'
    assert r.update(6) is None


@pytest.mark.parametrize('kw', [{'accuracy':float('nan')},{'accuracy':50},{'valid':False}])
def test_bad_fix_never_confirms(country, kw):
    r=resolver(country)
    assert r.update(0,fix(country,0,**kw)) is None
    assert r.update(1,fix(country,1,**kw)) is None


def test_future_duplicate_and_long_sample_gap(country):
    r=resolver(country)
    assert r.update(0,fix(country,1)) is None
    assert r.update(1,fix(country,1)) is None
    assert r.update(2,fix(country,1)) is None  # duplicate cannot accumulate confirmation time
    assert r.update(5,fix(country,5)) is None
    assert r.update(5.5,fix(country,5.5)) is None
    assert r.update(6,fix(country,6)) == 'FR'


def test_holes_overlaps_and_missing_policy(country):
    b=borders(country)
    b.countries[0]['polygons'][0].append([[-.8,-.2],[-.2,-.2],[-.2,.2],[-.8,.2],[-.8,-.2]])
    assert b.country_at(0,-.5,margin_m=0) is None
    assert country.CountryResolver.from_settings(b,{}) is None
    with pytest.raises(ValueError):
        country.CountryPolicy(2,20,-1,1,1)


def test_corrupt_deflate_blocks_lookup_instead_of_crashing(country,tmp_path):
    import gzip
    p=tmp_path/'broken.gz'
    content=bytearray(gzip.compress(b'{"countries": []}'))
    content[10]=255
    p.write_bytes(content)
    assert country.OfflineBorders.load(p).countries==[]


def test_overlapping_countries_are_unknown(country):
    b=borders(country)
    duplicate=dict(b.countries[0],code='US')
    b.countries.append(duplicate)
    assert b.country_at(0,-.5,margin_m=100) is None


def test_invalid_incoming_timestamp_or_fix_revokes_fresh_country(country):
    for bad in (country.GpsFix(float('nan'),0,-.5,5,True),country.GpsFix(1,0,-.5,5,False)):
        r=resolver(country)
        r.update(0,fix(country,0));r.update(.5,fix(country,.5))
        assert r.update(1,fix(country,1))=='FR'
        assert r.update(1.1,bad) is None


@pytest.mark.parametrize('cc,lat,lon,want', [('FR',48.86,2.35,8.3333333333),
    ('GB',51.507,-.127,13.4112),('US',39.74,-104.99,13.4112)])
def test_packaged_inland_coordinates_to_country_unit_and_physical_limit(country,vision,cc,lat,lon,want):
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'patches/speed-limits-unit-normalization/country-borders.json.gz'
    borders=country.OfflineBorders.load(path)
    r=country.CountryResolver(borders,country.CountryPolicy(2,20,100,1,1))
    assert r.update(0,country.GpsFix(0,lat,lon,5,True)) is None
    assert r.update(1,country.GpsFix(1,lat,lon,5,True))==cc
    m=vision.SpeedLimitMemory()
    m.set_country(r.country)
    m.update(1,[30])
    assert m.update(1.2,[30])==pytest.approx(want)
