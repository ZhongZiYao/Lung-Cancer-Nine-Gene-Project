"""Query Zenodo for DeepGEM dataset record 13917850."""
import urllib.request
import json

try:
    req = urllib.request.urlopen(
        'https://zenodo.org/api/records/13917850', timeout=30)
    d = json.loads(req.read())
    md = d.get('metadata', {})
    print('TITLE:', md.get('title'))
    print()
    print('DESCRIPTION:')
    print(md.get('description', '')[:2000])
    print()
    print('FILES:')
    for f in d.get('files', []):
        sz = f.get('size', 0)
        key = f.get('key')
        link = f.get('links', {}).get('self', '')
        print(f'  {sz/1e9:8.2f} GB   {key}')
        print(f'             {link}')
    print()
    print('TAGS/KEYWORDS:')
    for kw in md.get('keywords', []):
        print(' -', kw)
except Exception as e:
    print('Net failed:', type(e).__name__, e)
