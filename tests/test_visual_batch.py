from pathlib import Path
from PIL import Image, ImageDraw
import pytest
from pixelcue.visual_batch import analyse_batch, VISUALISER_PROMPTS


def images(tmp_path):
    paths=[]
    for i in range(4):
        image=Image.new('RGB',(64,64),'black')
        ImageDraw.Draw(image).ellipse((5+i,5,50,50),outline='lime',width=3)
        path=tmp_path/f'{i}.png';image.save(path);paths.append(path)
    return paths


def test_models_loaded_once_each_in_model_major_order(tmp_path):
    events=[]
    class Session:
        revision='fixture';model_type='fixture'
        def __init__(self,name,checkpoint,**options):
            self.name=name;events.append(('load',name));assert options['allow_download'] is False
            assert Path(options['cache_dir']).is_relative_to(tmp_path/'models')
        def infer(self,image):
            events.append(('infer',self.name));return {'caption':'a green ring'}
        def close(self): events.append(('close',self.name))
    result=analyse_batch(images(tmp_path),session_factory=Session,models_dir=tmp_path/'models')
    assert events==[('load','clip')]+[('infer','clip')]*4+[('close','clip'),('load','blip')]+[('infer','blip')]*4+[('close','blip')]
    assert len(result['items'])==4
    assert all(len(x['analysis']['models'])==2 for x in result['items'])
    assert result['prompt_profile']=='visualiser'


def test_unavailable_model_preserves_numeric_results(tmp_path):
    def fail(*args,**kwargs): raise OSError('not cached')
    result=analyse_batch(images(tmp_path),session_factory=fail,models_dir=tmp_path/'models')
    assert all(item['analysis']['measurements'] for item in result['items'])
    assert all(m['status']=='unavailable' for item in result['items'] for m in item['analysis']['models'])


def test_tagger_requires_explicit_permission(tmp_path):
    class Forbidden:
        def extract_keywords(self,*args,**kwargs): raise AssertionError('must not run')
    result=analyse_batch(images(tmp_path),models=[],keyword_models=['smolvlm-256m'],
                         models_dir=tmp_path/'models',keyword_service=Forbidden())
    assert result['items'][0]['analysis']['models'][0]['status']=='unavailable'


def test_tagger_reuses_service(tmp_path):
    class Service:
        def __init__(self): self.calls=0
        def extract_keywords(self,path,*,model): self.calls+=1;return ['ring']
    service=Service()
    result=analyse_batch(images(tmp_path),models=[],keyword_models=['fixture'],
        allow_download=True,keyword_service=service,models_dir=tmp_path/'models')
    assert service.calls==4
    assert result['items'][0]['analysis']['models'][0]['result']['tags']==['ring']


@pytest.mark.parametrize('kwargs',[{'models':['invented']},{'device':'auto'}, {'prompt_profile':'bad'},
                                 {'models':['blip'],'model_paths':{'clip':'somewhere'}}])
def test_bad_model_settings_rejected(tmp_path,kwargs):
    with pytest.raises(ValueError): analyse_batch(images(tmp_path),models_dir=tmp_path/'models',**kwargs)


def test_missing_input_does_not_block_valid_input(tmp_path):
    result=analyse_batch(images(tmp_path)+[tmp_path/'missing.png'],models=[],models_dir=tmp_path/'models')
    assert result['items'][-1]['status']=='unavailable'
    assert result['items'][0]['status']=='completed'


def test_prompt_bank_does_not_assert_temporal_behaviour():
    text=' '.join(p for group in VISUALISER_PROMPTS.values() for p in group)
    assert all(w not in text for w in ['clockwise','pulsing','rotates','strobing','NSFW','ominous'])
