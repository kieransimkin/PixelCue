import hashlib
import json
import math

import cv2
import numpy as np
from PIL import Image
import pytest

from pixelcue.visual_sequence import analyse_sequences, brightness_period, frame_features, pair_motion


def texture():
    a = np.random.default_rng(27).integers(20, 230, (128, 128, 3), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    a[15:30, 35:70] = 255
    return a


def sequence(tmp_path, kind='cw', count=18, condition='test', offset=0):
    a = texture()
    records = []
    for i in range(count):
        rotation = -i*1.4 if kind == 'cw' else i*1.4 if kind == 'ccw' else 0
        scale = 1.012**i if kind == 'expand' else .988**i if kind == 'contract' else 1
        mat = cv2.getRotationMatrix2D((63.5, 63.5), rotation, scale)
        if kind == 'right': mat[0, 2] += i
        if kind == 'up': mat[1, 2] -= i
        b = cv2.warpAffine(a, mat, (128, 128), borderMode=cv2.BORDER_REFLECT)
        if kind == 'white': b[:] = 255
        path = tmp_path/f'{condition}-{kind}-{i}.png'
        Image.fromarray(b).save(path)
        records.append(dict(id=f'{condition}:{i}', condition_id=condition, seconds=offset+i/15,
                            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return records


@pytest.mark.parametrize('kind,label', [('cw','clockwise rotation'),('ccw','counterclockwise rotation'),
    ('expand','outward expansion'),('contract','inward contraction'),('right','rightward drift'),
    ('up','upward drift'),('still','visually stationary in the sampled window')])
def test_known_motion(tmp_path, kind, label):
    evidence = analyse_sequences(sequence(tmp_path, kind))
    labels = [m['label'] for m in evidence['conditions'][0]['summary']['motion']]
    assert label in labels, labels
    assert evidence['state'] == 'completed'
    json.dumps(evidence, allow_nan=False)
    assert 'path' not in evidence['frames'][0]
    if kind in ('cw','ccw'):
        assert not any('drift' in x for x in labels)


def test_sparse_frames_never_establish_rotation(tmp_path):
    rows = sequence(tmp_path, 'cw', 4)
    for row, t in zip(rows,[0,.5,1.5,3]): row['seconds']=t
    result = analyse_sequences(rows)
    assert result['conditions'][0]['summary']['motion_status']=='unknown'
    assert all(t['status']=='unavailable' for t in result['conditions'][0]['transitions'])


def test_white_not_caption_eligible_or_stationary(tmp_path):
    result=analyse_sequences(sequence(tmp_path,'white'))
    assert all(not f['features']['semantic_eligible'] for f in result['frames'])
    assert result['conditions'][0]['summary']['motion_status']=='unknown'


@pytest.mark.parametrize('colour', ['white','black','red','#888888'])
def test_flat_fields_are_kept_but_not_captioned(colour):
    f=frame_features(Image.new('RGB',(80,80),colour))
    assert f['semantic_eligible'] is False
    assert len(f['descriptor'])==192


def test_no_motion_across_conditions_or_missing_frames(tmp_path):
    rows=sequence(tmp_path,'right',8,'a')+sequence(tmp_path,'ccw',8,'b')
    rows[3]['sha256']='0'*64
    result=analyse_sequences(rows)
    assert len(result['errors'])==1
    for c in result['conditions']:
        for pair in c['transitions']:
            assert pair['a'].startswith(c['id']+':') and pair['b'].startswith(c['id']+':')
            assert pair['id'] != 'a:2->a:4'
    assert result['state']=='partial'


@pytest.mark.parametrize('damage', ['duplicate_id','duplicate_time','nan','bad_hash'])
def test_invalid_record_rejected(tmp_path,damage):
    rows=sequence(tmp_path,'still',2)
    if damage=='duplicate_id': rows[1]['id']=rows[0]['id']
    if damage=='duplicate_time': rows[1]['seconds']=0
    if damage=='nan': rows[1]['seconds']=float('nan')
    if damage=='bad_hash': rows[1]['sha256']='abc'
    with pytest.raises(ValueError): analyse_sequences(rows)


def luminosities(values):
    return [dict(seconds=i/15,features={'mean_linear_luminance':v}) for i,v in enumerate(values)]


def test_brightness_pulsing_and_fade_distinguished():
    t=np.arange(61)/15
    pulse=brightness_period(luminosities(.5+.3*np.sin(2*np.pi*1.5*t)))
    assert pulse['status']=='measured'
    assert pulse['frequency_hz']==pytest.approx(1.5,abs=.25)
    assert brightness_period(luminosities(np.linspace(.1,.9,61)))['status']!='measured'


def test_irregular_period_not_interpolated():
    rows=luminosities([.3,.8]*32)
    rows[20]['seconds']+=.02
    assert brightness_period(rows)['status']=='unknown'


def test_dimension_change_and_zero_dt():
    a=Image.fromarray(texture())
    assert pair_motion(a,a.resize((80,80)),.1)['status']=='unavailable'
    with pytest.raises(ValueError): pair_motion(a,a,0)
