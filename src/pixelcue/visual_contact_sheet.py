"""Checksum-checked contact sheets and exact retained keyframes, without models."""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import shutil

from PIL import Image, ImageDraw, ImageFont


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_contact_sheets(records, groups, output, *, columns=4, tile=192):
    """Groups contain title and frame IDs. Exact PNG bytes are retained once.

    A sheet is a labelled preview, not a substitute for its full-quality source.
    Missing frames are visible placeholders and explicit errors, never omitted.
    """
    if not 1 <= columns <= 8 or not 64 <= tile <= 512 or len(groups) > 512:
        raise ValueError('Contact sheet layout exceeds bounds')
    by_id = {r['id']: r for r in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate frame identities')
    out = Path(output)
    keyframes, previews = out/'keyframes', out/'contact-sheets'
    keyframes.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    exports, sheets, errors = {}, [], []
    font = ImageFont.load_default(size=13)
    for group_no, group in enumerate(groups):
        identities = list(dict.fromkeys(group['ids']))
        # Paging limits memory, and includes ALL requested semantic frames.
        for page_no, offset in enumerate(range(0, len(identities), 16)):
            ids = identities[offset:offset+16]
            rows = (len(ids)+columns-1)//columns
            sheet = Image.new('RGB', (columns*(tile+12)+12, rows*(tile+60)+56), '#101925')
            draw = ImageDraw.Draw(sheet)
            title = str(group.get('title', 'Captured output'))[:130]
            draw.text((12, 12), title, fill='white', font=font)
            for index, identity in enumerate(ids):
                x, y = 12+(index%columns)*(tile+12), 45+(index//columns)*(tile+60)
                row = by_id.get(identity)
                try:
                    if row is None or not re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')):
                        raise ValueError('Unknown frame ID/hash')
                    source = Path(row['path'])
                    if file_hash(source) != row['sha256']:
                        raise ValueError('Capture checksum mismatch')
                    # All renderer captures in this protocol are PNG. Verify format
                    # before copying to a content-addressed .png filename.
                    with Image.open(source) as im:
                        if im.format != 'PNG':
                            raise ValueError('A lossless PNG capture is required')
                        if im.width*im.height > 16_777_216:
                            raise ValueError('Oversized capture')
                        im.load()
                        rgb = Image.new('RGBA', im.size, 'black')
                        rgb.alpha_composite(im.convert('RGBA'))
                        thumb = rgb.convert('RGB')
                        thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
                        sheet.paste(thumb, (x+(tile-thumb.width)//2, y+(tile-thumb.height)//2))
                        rgb.close(); thumb.close()
                    target = keyframes/(row['sha256']+'.png')
                    if target.exists():
                        if file_hash(target) != row['sha256']:
                            raise ValueError('Retained keyframe checksum mismatch')
                    else:
                        shutil.copyfile(source, target)
                    if file_hash(target) != row['sha256']:
                        raise ValueError('Retained keyframe checksum mismatch')
                    exports[identity] = {k: row[k] for k in ('id','condition_id','seconds','sha256')}
                    exports[identity]['file'] = target.relative_to(out).as_posix()
                    if 'pcm_rms' in row:
                        exports[identity]['pcm_rms'] = row['pcm_rms']
                except (OSError, ValueError) as exc:
                    draw.rectangle((x, y, x+tile, y+tile), outline='#dd8899', width=2)
                    draw.text((x+8, y+tile//2), 'CAPTURE UNAVAILABLE', fill='white', font=font)
                    errors.append({'id': identity, 'reason': type(exc).__name__+': '+str(exc).split(':')[0]})
                label = 'missing frame' if row is None else f"{row['seconds']:.3f} s | {row['condition_id']}"
                draw.text((x, y+tile+4), label[:27], fill='white', font=font)
                draw.text((x, y+tile+23), label[27:54], fill='#b8cde2', font=font)
            path = previews/f'sheet-{group_no:03}-{page_no:02}.webp'
            sheet.save(path, 'WEBP', lossless=True)
            sheet.close()
            sheets.append({'file': path.relative_to(out).as_posix(), 'sha256': file_hash(path),
                           'title': title, 'frame_ids': ids, 'kind': group.get('kind', 'condition')})
    return {'schema': 'pixelcue.contact-sheets/1', 'sheets': sheets,
            'keyframes': sorted(exports.values(), key=lambda r: r['id']), 'errors': errors,
            'retention': 'Exact selected PNG bytes; sheets are resized, labelled previews, not numerical evidence.'}
