\
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image


def load_cached_thumbnail(path: Path, requested_size: int) -> Image.Image | None:
    """Retrieve an existing Windows Shell thumbnail without generating one."""
    if sys.platform != "win32":
        return None

    try:
        import ctypes
        from ctypes import POINTER, Structure, byref, cast, c_void_p, c_wchar_p
        from ctypes.wintypes import BYTE, DWORD, HANDLE, HBITMAP, LONG, UINT, WORD
        from comtypes import COMMETHOD, GUID, HRESULT, IUnknown, CoInitialize, CoUninitialize
    except Exception:
        return None

    class SIZE(Structure):
        _fields_ = [("cx", LONG), ("cy", LONG)]

    class BITMAP(Structure):
        _fields_ = [
            ("bmType", LONG), ("bmWidth", LONG), ("bmHeight", LONG),
            ("bmWidthBytes", LONG), ("bmPlanes", WORD), ("bmBitsPixel", WORD),
            ("bmBits", c_void_p),
        ]

    class BITMAPINFOHEADER(Structure):
        _fields_ = [
            ("biSize", DWORD), ("biWidth", LONG), ("biHeight", LONG),
            ("biPlanes", WORD), ("biBitCount", WORD), ("biCompression", DWORD),
            ("biSizeImage", DWORD), ("biXPelsPerMeter", LONG),
            ("biYPelsPerMeter", LONG), ("biClrUsed", DWORD),
            ("biClrImportant", DWORD),
        ]

    class RGBQUAD(Structure):
        _fields_ = [
            ("rgbBlue", BYTE), ("rgbGreen", BYTE), ("rgbRed", BYTE),
            ("rgbReserved", BYTE),
        ]

    class BITMAPINFO(Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD * 1)]

    class IShellItemImageFactory(IUnknown):
        _case_insensitive_ = True
        _iid_ = GUID("{bcc18b79-ba16-442f-80c4-8a59c30c463b}")
        _idlflags_ = []

    IShellItemImageFactory._methods_ = [
        COMMETHOD([], HRESULT, "GetImage", (["in"], SIZE, "size"),
                  (["in"], UINT, "flags"), (["out"], POINTER(HBITMAP), "phbm"))
    ]

    shell32 = ctypes.windll.shell32
    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32

    shell32.SHCreateItemFromParsingName.argtypes = [c_wchar_p, c_void_p, POINTER(GUID), POINTER(c_void_p)]
    shell32.SHCreateItemFromParsingName.restype = HRESULT
    gdi32.GetObjectW.argtypes = [HANDLE, ctypes.c_int, c_void_p]
    gdi32.GetObjectW.restype = ctypes.c_int
    gdi32.GetDIBits.argtypes = [HANDLE, HANDLE, UINT, UINT, c_void_p, POINTER(BITMAPINFO), UINT]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [HANDLE]
    gdi32.DeleteObject.restype = ctypes.c_int
    user32.GetDC.argtypes = [HANDLE]
    user32.GetDC.restype = HANDLE
    user32.ReleaseDC.argtypes = [HANDLE, HANDLE]
    user32.ReleaseDC.restype = ctypes.c_int

    SIIGBF_THUMBNAILONLY = 0x00000008
    SIIGBF_INCACHEONLY = 0x00000010
    flags = SIIGBF_THUMBNAILONLY | SIIGBF_INCACHEONLY
    DIB_RGB_COLORS = 0
    BI_RGB = 0

    initialized = False
    hbitmap = None
    dc = None
    factory_ptr = c_void_p()
    try:
        CoInitialize()
        initialized = True
        hr = shell32.SHCreateItemFromParsingName(
            str(path).replace("/", "\\"), None,
            byref(IShellItemImageFactory._iid_), byref(factory_ptr),
        )
        if hr < 0 or not factory_ptr.value:
            return None

        factory = cast(factory_ptr, POINTER(IShellItemImageFactory))
        hbitmap = factory.GetImage(SIZE(requested_size, requested_size), flags)
        if not hbitmap:
            return None

        bitmap = BITMAP()
        if not gdi32.GetObjectW(hbitmap, ctypes.sizeof(bitmap), byref(bitmap)):
            return None
        width, height = int(bitmap.bmWidth), abs(int(bitmap.bmHeight))
        if width <= 0 or height <= 0:
            return None

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        raw = ctypes.create_string_buffer(width * height * 4)
        dc = user32.GetDC(None)
        if not dc:
            return None
        lines = gdi32.GetDIBits(dc, hbitmap, 0, height, raw, byref(bmi), DIB_RGB_COLORS)
        if lines != height:
            return None
        return Image.frombuffer("RGBA", (width, height), raw.raw, "raw", "BGRA", 0, 1).copy()
    except Exception:
        return None
    finally:
        if dc:
            try: user32.ReleaseDC(None, dc)
            except Exception: pass
        if hbitmap:
            try: gdi32.DeleteObject(hbitmap)
            except Exception: pass
        if initialized:
            try: CoUninitialize()
            except Exception: pass
