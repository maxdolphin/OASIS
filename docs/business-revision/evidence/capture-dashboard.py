#!/usr/bin/env python3
"""Capture full-page dashboard screenshots from the live OASIS Streamlit app via CDP.

Drives http://localhost:8501/ in a headless Chrome (remote-debugging-port=9222):
  1. Selects the "Use Sample Data" radio in the Control Panel.
  2. Clicks the Analyze button for the target org.
  3. Iterates every "Analysis Sections" radio option, screenshotting each full page.

Usage:
    python3 capture-dashboard.py "<org sidebar label>" <filename-prefix>

Requires: websocket-client, a running Streamlit app on :8501, Chrome CDP on :9222.
"""
import base64
import json
import os
import re
import sys
import time

import websocket

CDP_URL = "http://localhost:9222/json"
APP_URL = "http://localhost:8501/"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboards")

# Timings (seconds). Bumped generously; Streamlit renders over the wire in real time.
WAIT_FIRST_RENDER = 12
WAIT_AFTER_SAMPLE = 7
WAIT_AFTER_ANALYZE = 22
WAIT_PER_SECTION = 8


def get_page_ws():
    import urllib.request

    targets = json.loads(urllib.request.urlopen(CDP_URL).read())
    page = next((t for t in targets if t.get("type") == "page"), None)
    if page is None:
        raise RuntimeError("No page target found in CDP")
    return websocket.create_connection(
        page["webSocketDebuggerUrl"],
        max_size=None,
        timeout=120,
        header=["Origin: http://localhost:9222"],
    )


mid = 0


def make_cmd(ws):
    def cmd(m, p=None):
        global mid
        mid += 1
        ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
        while True:
            r = json.loads(ws.recv())
            if r.get("id") == mid:
                return r

    return cmd


def slugify(s):
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "section"


def main():
    if len(sys.argv) < 3:
        print("Usage: capture-dashboard.py '<org label>' <prefix>")
        sys.exit(2)
    org_label = sys.argv[1]
    prefix = sys.argv[2]
    os.makedirs(OUT_DIR, exist_ok=True)

    ws = get_page_ws()
    cmd = make_cmd(ws)

    def ev(expr):
        r = cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return r.get("result", {}).get("result", {}).get("value")

    cmd("Page.enable")
    cmd("Runtime.enable")
    cmd("DOM.enable")

    print(f"[nav] {APP_URL}")
    cmd("Page.navigate", {"url": APP_URL})
    time.sleep(WAIT_FIRST_RENDER)

    # 1. Click the "Use Sample Data" radio label.
    clicked = ev(
        r"""
        (function(){
          var labels = Array.from(document.querySelectorAll('label'));
          var t = labels.find(function(l){ return /use sample/i.test(l.innerText||''); });
          if(!t) return 'NO_SAMPLE_LABEL';
          t.click();
          return 'OK';
        })()
        """
    )
    print(f"[sample-data] {clicked}")
    time.sleep(WAIT_AFTER_SAMPLE)

    # 2. Find and click the Analyze button whose ancestor container mentions the org.
    org_js = json.dumps(org_label)
    analyze_result = ev(
        r"""
        (function(){
          var target = %s;
          var btns = Array.from(document.querySelectorAll('button'));
          var analyzeBtns = btns.filter(function(b){ return /analyze/i.test(b.innerText||''); });
          if(analyzeBtns.length===0) return 'NO_ANALYZE_BUTTONS';
          // For each Analyze button, find the SMALLEST (nearest) ancestor whose text
          // contains the target org label. The button whose nearest-matching ancestor
          // is the tightest (shortest text) is the correct card, because a very high
          // ancestor contains ALL org labels and would false-match every button.
          function nearestMatchLen(el){
            var node = el;
            for(var i=0;i<12 && node;i++){
              node = node.parentElement;
              if(!node) break;
              var t = node.innerText||'';
              if(t.indexOf(target)>=0) return t.length;  // first (nearest) ancestor to match
            }
            return Infinity;
          }
          var best=null, bestLen=Infinity;
          analyzeBtns.forEach(function(b){
            var L = nearestMatchLen(b);
            if(L < bestLen){ bestLen=L; best=b; }
          });
          if(!best || bestLen===Infinity) return 'NO_MATCH_FOR_ORG';
          best.scrollIntoView();
          best.click();
          return 'OK:'+analyzeBtns.length+' analyze buttons; matched card text len='+bestLen;
        })()
        """
        % org_js
    )
    print(f"[analyze] {analyze_result}")
    time.sleep(WAIT_AFTER_ANALYZE)

    # Confirm the org name appears in the results header area.
    org_present = ev(
        "(function(){var t=document.body.innerText||''; return t.indexOf(%s)>=0;})()" % org_js
    )
    print(f"[verify-org-in-page] {org_present}")

    # 3. Enumerate the "Analysis Sections" radio options.
    sections = ev(
        r"""
        (function(){
          var labels = Array.from(document.querySelectorAll('label'));
          // Sidebar radio group for analysis sections: labels that look like section names.
          // Collect radio-group option labels by finding the group headed near "Analysis Section".
          var texts = labels.map(function(l){return (l.innerText||'').trim();}).filter(Boolean);
          return JSON.stringify(texts);
        })()
        """
    )
    all_label_texts = json.loads(sections) if sections else []
    # Known analysis section option labels (match loosely against what's present).
    KNOWN = [
        "Core Metrics", "Network Analysis", "Visualizations",
        "OASIS Health", "Detailed Report", "Analysis Report",
        "System Health", "Overview",
    ]
    found_sections = []
    for txt in all_label_texts:
        for k in KNOWN:
            if k.lower() in txt.lower() and txt not in found_sections:
                found_sections.append(txt)
    # Deduplicate preserving order.
    seen = set()
    found_sections = [x for x in found_sections if not (x in seen or seen.add(x))]
    print(f"[sections-found] {found_sections}")
    print(f"[all-labels] {all_label_texts}")

    if not found_sections:
        print("[warn] no analysis-section labels matched; capturing single full page")
        found_sections = ["__single__"]

    saved = []
    for sec in found_sections:
        if sec != "__single__":
            sec_js = json.dumps(sec)
            click_res = ev(
                r"""
                (function(){
                  var target = %s;
                  var labels = Array.from(document.querySelectorAll('label'));
                  var l = labels.find(function(x){ return (x.innerText||'').trim()===target; });
                  if(!l) l = labels.find(function(x){ return (x.innerText||'').trim().indexOf(target)>=0; });
                  if(!l) return 'NO_LABEL';
                  l.scrollIntoView();
                  l.click();
                  return 'OK';
                })()
                """
                % sec_js
            )
            print(f"[section:{sec}] click={click_res}")
            time.sleep(WAIT_PER_SECTION)
            # scroll back to top for a clean full-page capture
            ev("window.scrollTo(0,0)")
            time.sleep(1)

        shot = cmd(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": True, "fromSurface": True},
        )
        data = shot.get("result", {}).get("data")
        if not data:
            print(f"[section:{sec}] SCREENSHOT FAILED: {shot}")
            continue
        slug = slugify(sec) if sec != "__single__" else "full"
        fname = os.path.join(OUT_DIR, f"{prefix}-{slug}.png")
        with open(fname, "wb") as fh:
            fh.write(base64.b64decode(data))
        size = os.path.getsize(fname)
        saved.append((fname, size))
        print(f"[saved] {fname} ({size} bytes)")

    ws.close()
    print("\n=== SUMMARY ===")
    print(f"org: {org_label}  prefix: {prefix}")
    print(f"sections: {found_sections}")
    for f, s in saved:
        print(f"  {f}  {s} bytes")


if __name__ == "__main__":
    main()
