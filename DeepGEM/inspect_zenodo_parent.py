"""Inspect Zenodo parent record + DeepGEM.zip details."""
import urllib.request
import json

print('=' * 80)
print('Zenodo parent record (13917849)')
print('=' * 80)
req = urllib.request.urlopen('https://zenodo.org/api/records/13917849', timeout=30)
d = json.loads(req.read())
md = d.get('metadata', {})
print('Title     :', md.get('title'))
print('Pub date  :', md.get('publication_date'))
print('Access    :', md.get('access_right'))
print('Resource  :', md.get('resource_type'))
print()
print('Description:')
desc = md.get('description', '')
print(desc[:3000])
print()
print('Files:')
for f in d.get('files', []):
    sz = f.get('size', 0)
    key = f.get('key')
    print(f'  {sz/1e9:8.2f} GB   {key}')
