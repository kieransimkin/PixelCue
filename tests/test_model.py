from pixelcue.model import parse_tags


def test_parse_special_tags():
    tags = parse_tags("cat, portrait, nsfw, face_identity, cat")
    assert tags == ["cat", "portrait", "NSFW", "FaceIdentity"]


def test_parse_sfw():
    tags = parse_tags("SFW, outdoor, blue sky")
    assert tags[0] == "SFW"


def test_parse_face_identity_with_space():
    tags = parse_tags("portrait, face identity, SFW")
    assert "FaceIdentity" in tags
