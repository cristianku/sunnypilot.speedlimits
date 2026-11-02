import pytest
from nnslr_tools.training import scheme_label
from nnslr_tools.exporting import reader_io_contract
from nnslr_tools.training import off_display


def row(**kw):
    return dict(review_state='accepted', bbox_xyxy=[1,2,3,4], sign_family='maximum_speed',
                value_state='read', **kw)


def test_mph_digits_never_become_converted_class():
    assert scheme_label(row(value_kph=None, sign_value=25, sign_unit='mph'), 'v2') == 'maximum_speed:25'
    assert scheme_label(row(value_kph=30), 'v2') == 'maximum_speed:30'


def test_conflicting_numeric_alias_is_rejected():
    with pytest.raises(ValueError, match='sign_value'):
        scheme_label(row(value_kph=40, sign_value=25, sign_unit='mph'), 'v2')


def test_new_contract_declares_written_number():
    c = reader_io_contract({'classes':['maximum_speed:25'], 'input_size':160})
    assert c['output']['number_semantics'] == 'written_sign_value'
    assert c['output']['unit_source'] == 'runtime_country'


def test_lit_variable_display_with_new_value_is_not_off():
    r=row(value_kph=None,sign_value=25,sign_unit='mph')
    r.update(sign_family='variable_display',notes='number, other panel off')
    assert not off_display(r)
