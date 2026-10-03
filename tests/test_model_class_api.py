from pixelcue.model import JoyCaption4Bit


def test_joycaption_public_methods_are_real_class_methods():
    assert callable(getattr(JoyCaption4Bit, "ensure_downloaded", None))
    assert callable(getattr(JoyCaption4Bit, "load", None))
    assert callable(getattr(JoyCaption4Bit, "tags_for_image", None))


def test_joycaption_methods_are_defined_directly_on_class():
    assert "ensure_downloaded" in JoyCaption4Bit.__dict__
    assert "load" in JoyCaption4Bit.__dict__
    assert "tags_for_image" in JoyCaption4Bit.__dict__
