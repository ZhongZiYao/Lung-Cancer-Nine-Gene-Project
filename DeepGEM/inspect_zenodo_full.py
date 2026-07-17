"""Inspect Zenodo full metadata."""
import urllib.request, json

req = urllib.request.urlopen('https://zenodo.org/api/records/13917849', timeout=30)
d = json.loads(req.read())
md = d['metadata']

print('Full description:')
print(md.get('description'))
print()
print('Related identifiers:')
for r in md.get('related_identifiers', []):
    print('  ', r.get('relation'), ':', r.get('identifier'))
print()
print('Resource type     :', md.get('resource_type'))
print('Access            :', md.get('access_right'))
print('License           :', md.get('license'))
print('Creators          :')
for c in md.get('creators', []):
    print(' -', c.get('name'), '|', c.get('affiliation'))
