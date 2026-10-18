import hashlib
import numpy as np
from PIL import Image
import pytest
from pixelcue.visual_sequence import analyse_sequences, pair_motion
from pixelcue.visual_motion_detail import local_flow
from pixelcue.visual_contact_sheet import export_contact_sheets


def texture(edge=128):
    import cv2
    rng=np.random.default_rng(213)
    return cv2.GaussianBlur(rng.integers(20,235,(edge,edge),dtype=np.uint8),(3,3),0)


def records_for(tmp_path, arrays, fps=15):
    rows=[]
    for i,array in enumerate(arrays):
        path=tmp_path/f'f-{i}.png';Image.fromarray(array).convert('RGB').save(path)
        rows.append({'id':str(i),'condition_id':'test','seconds':i/fps,'path':str(path),
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pcm_rms':.01*i,'frame_index':i})
    return rows


def test_slow_contraction_uses_longer_baseline_and_keeps_all_frames(tmp_path):
    import cv2
    a=texture()
    arrays=[cv2.warpAffine(a,cv2.getRotationMatrix2D((63.5,63.5),0,np.exp(-.022*i/15)),(128,128)) for i in range(76)]
    rows=records_for(tmp_path,arrays)
    result=analyse_sequences(rows,max_edge=128,multi_lag=True)
    assert len(result['frames'])==76
    assert all('pcm_rms' in f for f in result['frames'])
    c=result['conditions'][0]
    assert len(c['transitions'])==75
    assert len(c['lagged_transitions'])>=6
    assert 'slow inward contraction' in [x['label'] for x in c['summary']['motion']]


def test_independent_local_flow_does_not_need_a_global_similarity():
    import cv2
    a=texture(192);b=np.zeros_like(a)
    for i,(dx,dy) in enumerate(((3,0),(-3,0),(0,3),(0,-3))):
        y,x=(i//2)*96,(i%2)*96
        b[y:y+96,x:x+96]=cv2.warpAffine(a[y:y+96,x:x+96],np.float32([[1,0,dx],[0,1,dy]]),(96,96))
    result=pair_motion(Image.fromarray(a),Image.fromarray(b),1/15)
    assert result['status']=='unavailable'
    assert result['local']['status']=='measured'
    assert result['local']['dispersion_per_second']>.008
    assert 'rotation_radians_per_second' not in result


def test_global_exposure_change_does_not_fabricate_local_motion():
    a=texture()
    b=np.clip(a.astype(float)*.5+30,0,255).astype(np.uint8)
    r=local_flow(Image.fromarray(a),Image.fromarray(b),1/15)
    assert r['status']=='unavailable' or r['speed_per_second']<.008


def test_blank_frames_are_not_stationary_motion_evidence(tmp_path):
    rows=records_for(tmp_path,[np.zeros((64,64),np.uint8)]*15)
    result=analyse_sequences(rows,max_edge=64,multi_lag=True)
    assert not result['conditions'][0]['summary']['motion']


def test_original_pngs_and_semantic_selections_are_retained(tmp_path):
    rows=records_for(tmp_path,[texture(64)]*3)
    groups=[{'title':'sound trial','ids':['0','1'],'kind':'condition'},
            {'title':'model selections','ids':['2'],'kind':'semantic'}]
    r=export_contact_sheets(rows,groups,tmp_path/'report')
    assert len(r['sheets'])==2 and len(r['keyframes'])==3 and not r['errors']
    for row in r['keyframes']:
        target=tmp_path/'report'/row['file']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
        assert Image.open(target).size==(64,64)
    assert (tmp_path/'report'/r['sheets'][0]['file']).is_file()


def test_missing_capture_is_visible_not_silently_omitted(tmp_path):
    rows=records_for(tmp_path,[texture(64)])
    (tmp_path/'f-0.png').unlink()
    r=export_contact_sheets(rows,[{'ids':['0']}],tmp_path/'out')
    assert len(r['errors'])==1 and len(r['sheets'])==1 and not r['keyframes']


def test_sheets_page_all_requested_frames(tmp_path):
    rows=records_for(tmp_path,[texture(64)]*33)
    r=export_contact_sheets(rows,[{'ids':[r['id'] for r in rows]}],tmp_path/'out')
    assert len(r['sheets'])==3
    assert sum(len(s['frame_ids']) for s in r['sheets'])==33


def test_changed_hash_never_becomes_a_retained_original(tmp_path):
    rows=records_for(tmp_path,[texture(64)]);rows[0]['sha256']='0'*64
    r=export_contact_sheets(rows,[{'ids':['0']}],tmp_path/'out')
    assert r['errors'] and not r['keyframes']


@pytest.mark.parametrize('dt',[0,-1,float('nan')])
def test_invalid_intervals_rejected(dt):
    a=Image.fromarray(texture(64))
    with pytest.raises(ValueError): pair_motion(a,a,dt)
