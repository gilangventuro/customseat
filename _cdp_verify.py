import json, time, base64, sys, urllib.request
import websocket

port = sys.argv[1] if len(sys.argv) > 1 else "9999"
url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8732/dummy-kursi-twinseat.html"
out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/verify.png"
wait_s = float(sys.argv[4]) if len(sys.argv) > 4 else 6

tabs = json.loads(urllib.request.urlopen(f"http://localhost:{port}/json").read())
tab = tabs[0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=45)

def send(method, params=None, mid=[1]):
    msg = {"id": mid[0], "method": method, "params": params or {}}
    mid[0] += 1
    ws.send(json.dumps(msg))
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == msg["id"]:
            return resp

send("Page.enable")
send("Page.navigate", {"url": url})
time.sleep(wait_s)
r = send("Runtime.evaluate", {"expression": "document.title"})
print("title:", r["result"]["result"]["value"])

for attempt in range(3):
    try:
        r = send("Page.captureScreenshot", {"format": "png"})
        data = base64.b64decode(r["result"]["data"])
        with open(out, "wb") as f:
            f.write(data)
        print("saved", out, len(data))
        break
    except Exception as e:
        print("attempt", attempt, "failed:", e)
        time.sleep(2)
ws.close()
