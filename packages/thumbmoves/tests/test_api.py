from thumbmoves import backend_info
from thumbmoves.common import normalize_size


def test_backend_info_is_cross_platform_api():
    assert backend_info("win32").name == "windows-shell"
    assert backend_info("win32").cache_only_supported is True
    assert backend_info("darwin").name == "macos-quicklook"
    assert backend_info("darwin").cache_only_supported is False
    assert backend_info("linux").name == "freedesktop-xdg"


def test_size_validation():
    assert normalize_size((320, 240)) == (320, 240)
    try:
        normalize_size((0, 240))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
