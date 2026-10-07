import requests, json
S = requests.Session(); S.trust_env = False
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
for u in ["https://hf-mirror.com/api/whoami-v2", "https://huggingface.co/api/whoami-v2"]:
    try:
        r = S.get(u, headers=H, timeout=30)
        print("---", u, "->", r.status_code, flush=True)
        if r.status_code == 200:
            j = r.json()
            print("   name :", j.get("name"), flush=True)
            print("   email:", j.get("email"), flush=True)
            print("   type :", j.get("type"), flush=True)
            print("   full :", json.dumps(j)[:400], flush=True)
        else:
            print("   ", r.text[:150], flush=True)
    except Exception as e:
        print("---", u, "ERR", str(e)[:80], flush=True)