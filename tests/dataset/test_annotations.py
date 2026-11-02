# [reviewed-dataset] - START
from __future__ import annotations
import copy
import hashlib
import json
import struct
import zlib
from pathlib import Path
import pytest
from nnslr_tools import cli


def png(path, value=30):
    def chunk(t, data):
        return struct.pack(">I",len(data))+t+data+struct.pack(">I",zlib.crc32(t+data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",32,24,8,2,0,0,0))+chunk(b"IDAT",zlib.compress((b"\0"+bytes([value,0,0])*32)*24))+chunk(b"IEND",b""))


def reviewed(root, route="00000001--abcdef1234", *, index=0, negative=False):
    image=root/"derived/frames"/route/"0"/f"frame_{index:08d}.png"
    png(image,int(route[:8],16)%256)
    detection=dict(detection_id="box-0",bbox_xyxy=[1,2,15,20],family="maximum_speed",value_kph=30,
                   value_state="reviewed",score=.8,review_state="reviewed",applicability="unresolved",
                   supplementary_panel="unknown")
    return dict(schema_version=1,kind="reviewed_frame",route_id=route,segment_index=0,output_index=index,
                decoded_frame_index=None,capture_mono_ns=None,alignment_status="unresolved",
                camera_stream="narrow_road",image_path=str(image.relative_to(root)),
                frame_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),width=32,height=24,
                review_state="reviewed",reviewer="test-human",review_timestamp="2026-09-19T12:00:00Z",
                label_basis="human_review",ground_truth=False,training_ready=False,run_id="synthetic",
                detections=[] if negative else [detection],source_proposals=[] if negative else [copy.deepcopy(detection)])


def invoke(capsys,*args):
    try: code=cli.main(list(args))
    except SystemExit as e: code=e.code
    output=capsys.readouterr()
    return code,output


def write(root, rows):
    source=root/"reviews.jsonl"
    source.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return source


def test_original_mph_number_survives_review_and_canonicalization(tmp_path):
    from nnslr_tools.annotations import reviewed_frame_annotations, validate_annotation
    from nnslr_tools.review_db import _reviewed_detection
    frame = reviewed(tmp_path)
    frame['country'] = 'US'
    d = dict(frame['detections'][0], value_kph=None, sign_value=25, sign_unit='mph')
    frame['detections'] = [_reviewed_detection(d, {})]
    rows = reviewed_frame_annotations(frame, 'a' * 64)
    row = rows[0]
    assert (row['value_kph'], row['sign_value'], row['sign_unit'], row['value_state']) == (25,25,'mph','read')
    validate_annotation(row, tmp_path)
    from nnslr_tools.annotations import DatasetError
    row['value_kph'] = None
    with pytest.raises(DatasetError, match='sign_value_legacy_alias_missing'):
        validate_annotation(row,tmp_path)


@pytest.mark.parametrize('changes', [dict(sign_value=25, value_kph=40),
                                    dict(sign_value=True), dict(sign_value=25,sign_unit='knots')])
def test_invalid_original_number_metadata_rejected(tmp_path, changes):
    from nnslr_tools.annotations import reviewed_frame_annotations, DatasetError
    frame = reviewed(tmp_path)
    frame['detections'][0].update(changes)
    with pytest.raises(DatasetError):
        reviewed_frame_annotations(frame, 'a' * 64)


def test_import_reviewed_frame_default_output_and_negative(tmp_path, capsys):
    source=write(tmp_path,[reviewed(tmp_path), reviewed(tmp_path,"00000002--abcdef1234",negative=True)])
    code,out=invoke(capsys,"import-annotations",str(source),"--data-root",str(tmp_path))
    assert code==0,out.err+out.out
    records=[json.loads(s) for s in (tmp_path/"annotations/objects.jsonl").read_text().splitlines()]
    assert len(records)==2
    positive=next(r for r in records if r["sign_family"]=="maximum_speed")
    assert positive["value_kph"]==30 and positive["value_state"]=="read"
    assert positive["review_state"]=="accepted" and positive["reviewer"]=="test-human"
    assert positive["decoded_frame_index"] is None and positive["capture_mono_ns"] is None
    assert positive["provenance"]["source_review_sha256"]==hashlib.sha256(source.read_bytes()).hexdigest()
    assert next(r for r in records if r["sign_family"]=="not_a_sign")["bbox_xyxy"] is None
    code,out=invoke(capsys,"validate-dataset","--data-root",str(tmp_path))
    assert code==0,out.err+out.out
    report=json.loads(out.out)
    assert report["valid"] and report["timing_eligible_count"]==0


@pytest.mark.parametrize("field,value,reason",[
    ("review_state","pending","unreviewed_proposal"),
    ("reviewer",None,"missing_provenance"),
    ("review_timestamp","yesterday","invalid_review_timestamp"),
    ("width",33,"image_dimensions_mismatch"),
    ("frame_sha256","0"*64,"hash_mismatch"),
    ("capture_mono_ns",123,"capture_time_inconsistency"),
])
def test_invalid_reviews_leave_no_dataset(tmp_path,capsys,field,value,reason):
    row=reviewed(tmp_path);row[field]=value
    code,out=invoke(capsys,"import-annotations",str(write(tmp_path,[row])),"--data-root",str(tmp_path))
    assert code!=0
    assert reason in out.out+out.err
    assert not (tmp_path/"annotations/objects.jsonl").exists()


@pytest.mark.parametrize("field,value,reason",[
    ("bbox_xyxy",[-1,0,2,2],"invalid_bbox"),
    ("bbox_xyxy",[1,1,1,2],"invalid_bbox"),
    ("value_kph",0,"illegal_speed"),
    ("value_kph",True,"illegal_speed"),
    ("family","advertisement","unknown_enum"),
])
def test_invalid_detection_rejected(tmp_path,capsys,field,value,reason):
    row=reviewed(tmp_path);row["detections"][0][field]=value
    code,out=invoke(capsys,"import-annotations",str(write(tmp_path,[row])),"--data-root",str(tmp_path))
    assert code!=0 and reason in out.out+out.err


def test_duplicates_rejected_and_existing_dataset_preserved(tmp_path,capsys):
    source=write(tmp_path,[reviewed(tmp_path)])
    code,out=invoke(capsys,"import-annotations",str(source),"--data-root",str(tmp_path))
    assert code==0,out.err+out.out
    dataset=tmp_path/"annotations/objects.jsonl";original=dataset.read_bytes()
    code,out=invoke(capsys,"import-annotations",str(source),"--data-root",str(tmp_path))
    assert code!=0 and "duplicate_identity" in out.out+out.err
    assert dataset.read_bytes()==original


def test_modified_and_deleted_proposals_preserve_originals(tmp_path,capsys):
    row=reviewed(tmp_path);row["detections"][0]["value_kph"]=50
    source=write(tmp_path,[row])
    code,out=invoke(capsys,"import-annotations",str(source),"--data-root",str(tmp_path))
    assert code==0,out.err+out.out
    record=json.loads((tmp_path/"annotations/objects.jsonl").read_text())
    assert record["review_state"]=="corrected" and record["value_kph"]==50
    assert record["provenance"]["original_proposal"]["value_kph"]==30


def test_dataset_validation_detects_changed_and_missing_images(tmp_path,capsys):
    row=reviewed(tmp_path)
    code,out=invoke(capsys,"import-annotations",str(write(tmp_path,[row])),"--data-root",str(tmp_path))
    assert code==0,out.err+out.out
    image=tmp_path/row["image_path"];png(image,90)
    code,out=invoke(capsys,"validate-dataset","--data-root",str(tmp_path))
    assert code!=0 and "hash_mismatch" in out.out
    image.unlink()
    code,out=invoke(capsys,"validate-dataset","--data-root",str(tmp_path))
    assert code!=0 and "missing_image" in out.out
# [reviewed-dataset] - END

# [reviewed-dataset] - START
def test_rejected_proposals_are_audit_only_and_negative_requires_full_review(tmp_path,capsys):
    row=reviewed(tmp_path);row['detections']=[]
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[row])),'--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    records=[json.loads(s) for s in (tmp_path/'annotations/objects.jsonl').read_text().splitlines()]
    assert {r['review_state'] for r in records}=={'accepted','rejected'}
    assert next(r for r in records if r['review_state']=='rejected')['provenance']['original_proposal']['value_kph']==30


def test_extra_canonical_fields_and_path_traversal_rejected(tmp_path,capsys):
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[reviewed(tmp_path)])),'--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    path=tmp_path/'annotations/objects.jsonl';record=json.loads(path.read_text());record['unexpected']=True
    path.write_text(json.dumps(record)+'\n')
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path))
    assert code!=0 and 'invalid_schema' in out.out
    record.pop('unexpected');record['image_path']='../escape.png';path.write_text(json.dumps(record)+'\n')
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path))
    assert code!=0 and ('unsafe_image_path' in out.out or 'path_traversal' in out.out)


def test_nonfinite_and_duplicate_json_keys_rejected(tmp_path,capsys):
    source=tmp_path/'invalid.jsonl'
    for text in ['{"value":NaN}', '{"value":1,"value":2}']:
        source.write_text(text)
        code,out=invoke(capsys,'import-annotations',str(source),'--data-root',str(tmp_path))
        assert code!=0 and 'invalid_json' in out.out


def test_default_inbox_and_root_without_environment(tmp_path,monkeypatch,capsys):
    import nnslr_tools.preannotate as module
    monkeypatch.delenv('NNSLR_DATA_ROOT',raising=False)
    monkeypatch.setattr(module,'DEFAULT_DATA_ROOT',str(tmp_path))
    source=write(tmp_path,[reviewed(tmp_path)])
    inbox=tmp_path/'annotations/inbox';inbox.mkdir(parents=True);source.rename(inbox/'review.jsonl')
    code,out=invoke(capsys,'import-annotations')
    assert code==0,out.err+out.out
# [reviewed-dataset] - END

# [reviewed-dataset] - START
@pytest.mark.parametrize('field,value', [('route_id',[]),('segment_index',{}),('output_index',True),('camera_stream',[])])
def test_invalid_review_identity_is_reported_without_traceback(tmp_path,capsys,field,value):
    row=reviewed(tmp_path);row[field]=value
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[row])),'--data-root',str(tmp_path))
    assert code!=0 and 'invalid_identity' in out.out


@pytest.mark.parametrize('field,value', [('detection_id',[]),('family',{}),('bbox_xyxy',False)])
def test_invalid_detection_types_are_reported_without_traceback(tmp_path,capsys,field,value):
    row=reviewed(tmp_path);row['detections'][0][field]=value
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[row])),'--data-root',str(tmp_path))
    assert code!=0 and json.loads(out.out)['valid'] is False


def test_canonical_schema_vocabulary_matches_validator():
    from nnslr_tools.annotations import FIELDS,ENUMS,OPTIONAL_FIELDS
    schema=json.loads((Path(__file__).parents[2]/'schemas/annotation.schema.json').read_text())
    assert set(schema['required'])==FIELDS-OPTIONAL_FIELDS and set(schema['properties'])==FIELDS
    assert all(set(schema['properties'][k]['enum'])==values for k,values in ENUMS.items())


def ai(row, channel='mcp'):
    """The same reviewed frame confirmed by an AI reviewer through the REST API or MCP."""
    row.update(reviewer='ai/Claude',label_basis='ai_review',reviewer_kind='ai',review_channel=channel)
    return row


def test_ai_review_is_a_gold_review_and_keeps_its_provenance(tmp_path,capsys):
    rows=[ai(reviewed(tmp_path)),reviewed(tmp_path,"00000002--abcdef1234",negative=True)]
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,rows)),'--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    records=[json.loads(s) for s in (tmp_path/'annotations/objects.jsonl').read_text().splitlines()]
    machine=[r for r in records if r['route_id']=="00000001--abcdef1234"]
    person=[r for r in records if r['route_id']=="00000002--abcdef1234"]
    assert machine and all(r['reviewer']=='ai/Claude' and r['schema_version']==1 for r in machine)
    assert all({k:r['provenance'][k] for k in ('label_basis','reviewer_kind','review_channel')}
               =={'label_basis':'ai_review','reviewer_kind':'ai','review_channel':'mcp'} for r in machine)
    assert person and all(r['provenance']['label_basis']=='human_review' and 'reviewer_kind' not in r['provenance']
                          for r in person)
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    jsonschema=pytest.importorskip('jsonschema')
    schema=json.loads((Path(__file__).parents[2]/'schemas/annotation.schema.json').read_text())
    for record in records:
        jsonschema.validate(record,schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(machine[0],reviewer='Anna'),schema)


@pytest.mark.parametrize('change,reason',[
    (dict(review_channel=None),'missing_provenance'),
    (dict(reviewer_kind='human'),'missing_provenance'),
    (dict(reviewer='Claude'),'reviewer_kind_mismatch'),
    (dict(review_state='pending'),'unreviewed_proposal'),
    (dict(kind='preannotated_frame'),'unreviewed_proposal'),
])
def test_ai_review_needs_its_provenance_and_a_confirmation(tmp_path,capsys,change,reason):
    row=ai(reviewed(tmp_path));row.update(change)
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[row])),'--data-root',str(tmp_path))
    assert code!=0 and reason in out.out
    assert not (tmp_path/'annotations/objects.jsonl').exists()


def test_ai_label_basis_in_a_canonical_row_needs_an_ai_reviewer(tmp_path,capsys):
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[ai(reviewed(tmp_path))])),'--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    from nnslr_tools.annotations import load_jsonl,validate_dataset
    rows=load_jsonl(tmp_path/'annotations/objects.jsonl')
    assert validate_dataset(rows,tmp_path)['valid']
    assert validate_dataset(rows,tmp_path,dataset_kind='training-candidate')['valid']
    forged=[dict(r,reviewer='test-human') for r in rows]
    assert validate_dataset(forged,tmp_path)['errors'][0]['reason']=='reviewer_kind_mismatch'
    unlabelled=[dict(r,provenance=dict(r['provenance'],review_channel='email')) for r in rows]
    assert validate_dataset(unlabelled,tmp_path)['errors'][0]['reason']=='missing_provenance'
# [reviewed-dataset] - END

# [dataset-review] - START
def test_whole_frame_negative_conflicts_with_later_positive(tmp_path,capsys):
    negative=reviewed(tmp_path,negative=True)
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[negative])),'--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    path=tmp_path/'annotations/objects.jsonl';before=path.read_bytes()
    positive=reviewed(tmp_path)
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[positive])),'--data-root',str(tmp_path))
    assert code!=0 and 'contradictory_frame_labels' in out.out
    assert path.read_bytes()==before


def test_truncated_png_with_matching_hash_is_not_dataset_truth(tmp_path,capsys):
    row=reviewed(tmp_path);path=tmp_path/row['image_path'];path.write_bytes(path.read_bytes()[:24])
    row['frame_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[row])),'--data-root',str(tmp_path))
    assert code!=0 and 'invalid_image' in out.out
    assert not (tmp_path/'annotations/objects.jsonl').exists()
# [dataset-review] - END

# [dataset-review] - START
def test_same_decoded_frame_cannot_be_reimported_with_new_png_counter(tmp_path,capsys):
    row=reviewed(tmp_path);row.update(decoded_frame_index=20,source_video_sha256='a'*64)
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[row])),'--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    row['output_index']=99
    code,out=invoke(capsys,'import-annotations',str(write(tmp_path,[row])),'--data-root',str(tmp_path))
    assert code!=0 and 'duplicate_identity' in out.out
# [dataset-review] - END
