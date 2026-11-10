#!/usr/bin/env python3
"""The play site's permanent link (user's choice, 2026-11-09: GitHub Pages, $0).

    https://<owner>.github.io/<repo>/?token=...   ->   https://<current>.trycloudflare.com/?token=...

The repo (default Aditya-Gupta26/pr-play, public, GitHub Pages on its main branch) holds two files: index.html, the
forwarding page (play/play_redirect.html), and target.json, the current tunnel address. Each site job sets target.json
when its tunnel is up and marks it down when it ends (only if it still names that job: a newer job may have taken
over). The forwarding page checks /api/alive before sending anyone, so a stale address is never followed. Open link (the
owner's choice, 2026-11-09; players sign in with name + PIN): target.json also carries the site's access token,
which the page puts into the visitor's link, so https://<owner>.github.io/<repo>/ alone opens the game, and changing
the token (then `set` again) does not break anyone's link. `set --private` leaves the token out (visitors then need
a link with ?token=...). The owner key is never written here.

The key: a fine-grained GitHub token with Contents read/write on that one repo, in
$PR_ROOT/secrets/github_pages_token (made by the owner; never printed or logged here). Standard library only, so the
site job runs it on the host.

    python3 play/play_link.py url                                # the permanent base URL
    python3 play/play_link.py show                               # target.json now
    python3 play/play_link.py set https://<x>.trycloudflare.com --job 22442925
    python3 play/play_link.py down --job 22442925
    python3 play/play_link.py publish-page                       # (re)upload index.html
    python3 play/play_link.py enable-pages                       # turn Pages on (main, /), if the key may (else: repo Settings -> Pages)
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR_ROOT = os.environ.get("PR_ROOT", f"/scratch/{os.environ.get('USER', '')}/pufferroyale")
TOKEN_FILE = os.path.join(PR_ROOT, "secrets", "github_pages_token")
SITE_TOKEN_FILE = os.path.join(PR_ROOT, "secrets", "play_site_token")
SITE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
PAGE_FILE = os.path.join(ROOT, "play", "play_redirect.html")
TUNNEL_RE = re.compile(r"^https://[a-z0-9-]+\.trycloudflare\.com$")
REPO_RE = re.compile(r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")


def pages_url(repo):
    owner, name = repo.split("/")
    return f"https://{owner.lower()}.github.io/{name}/"


KEY_RE = re.compile(r"github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}|ghp_[A-Za-z0-9]{36}")


def read_token():
    """The key from TOKEN_FILE. A paste into `read -s` can bring terminal control codes along (seen 2026-11-09: 21
    escape sequences around the key, so GitHub answered 401): the one well-formed key in the file is used."""
    try:
        raw = open(TOKEN_FILE, errors="replace").read()
    except OSError:
        raise SystemExit(f"no GitHub key at {TOKEN_FILE} (see docs/HUMAN_PLAY.md, 'Permanent link')")
    keys = set(KEY_RE.findall(raw))
    if len(keys) != 1:
        raise SystemExit(f"{TOKEN_FILE} should hold exactly one GitHub key (github_pat_... or ghp_...); found {len(keys)}")
    return keys.pop()


def api(method, repo, path, token, body=None):
    """One call to the contents endpoint; (status, parsed JSON or None). The token goes only in the header."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "pufferroyale-play-link"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read() or b"null")
        except ValueError:
            msg = None
        return e.code, msg


def enable_pages(repo, token):
    """Turn GitHub Pages on for the repo (branch main, root). Needs the key's "Pages: Read and write" permission."""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/pages", method="POST",
                                 data=json.dumps({"source": {"branch": "main", "path": "/"}}).encode(), headers={
                                     "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "pufferroyale-play-link"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        if e.code == 409:                                  # already on
            return 409
        raise SystemExit(f"GitHub refused to turn Pages on (HTTP {e.code}): the key may not be allowed to manage Pages "
                         f"(GitHub's docs do not list what a fine-grained key needs). Turn it on in the repo instead: "
                         f"Settings -> Pages -> Deploy from a branch -> main, / (root) -> Save.")


def get_file(repo, path, token):
    """(bytes or None, sha or None)."""
    st, d = api("GET", repo, path, token)
    if st == 404:
        return None, None
    if st != 200 or not isinstance(d, dict):
        raise SystemExit(f"GitHub GET {path}: HTTP {st} {(d or {}).get('message', '') if isinstance(d, dict) else ''}")
    return base64.b64decode(d.get("content", "")), d.get("sha")


def put_file(repo, path, data, message, token):
    """Create or replace a file (retries on a conflicting concurrent update). Returns True if it changed."""
    for attempt in range(4):
        old, sha = get_file(repo, path, token)
        if old == data:
            return False
        body = {"message": message, "content": base64.b64encode(data).decode(), "branch": "main"}
        if sha:
            body["sha"] = sha
        st, d = api("PUT", repo, path, token, body)
        if st in (200, 201):
            return True
        if st in (409, 422, 500, 502, 503) and attempt < 3:
            time.sleep(2 * (attempt + 1))
            continue
        raise SystemExit(f"GitHub PUT {path}: HTTP {st} {(d or {}).get('message', '') if isinstance(d, dict) else ''}")
    return False


def current(repo, token):
    data, _ = get_file(repo, "target.json", token)
    try:
        return json.loads(data) if data else None
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("url", "show", "set", "down", "publish-page", "enable-pages"))
    ap.add_argument("tunnel", nargs="?", help="set: https://<x>.trycloudflare.com")
    ap.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--repo", default=os.environ.get("PAGES_REPO", "Aditya-Gupta26/pr-play"))
    ap.add_argument("--site-token-file", default=SITE_TOKEN_FILE, help="set: the site's access token (open link)")
    ap.add_argument("--private", action="store_true", help="set: leave the site token out of target.json")
    a = ap.parse_args()
    if not REPO_RE.match(a.repo):
        raise SystemExit(f"bad repo {a.repo!r}")
    if a.cmd == "url":
        print(pages_url(a.repo))
        return
    token = read_token()
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if a.cmd == "show":
        print(json.dumps(current(a.repo, token), indent=1))
    elif a.cmd == "set":
        url = (a.tunnel or "").rstrip("/")
        if not TUNNEL_RE.match(url):
            raise SystemExit(f"not a quick-tunnel address: {url!r}")
        body = {"status": "up", "url": url, "job": str(a.job), "since": now}
        if not a.private:
            try:
                site_token = open(a.site_token_file).read().strip()
            except OSError:
                raise SystemExit(f"no site token at {a.site_token_file} (or use --private)")
            if not SITE_TOKEN_RE.match(site_token):
                raise SystemExit(f"{a.site_token_file} does not hold a site token")
            body["token"] = site_token
        changed = put_file(a.repo, "target.json", (json.dumps(body, indent=1) + "\n").encode(),
                           f"site up: job {a.job}", token)
        print(f"[play_link] {pages_url(a.repo)} -> {url} (job {a.job}, {'private' if a.private else 'open'} link)"
              f"{'' if changed else ' (unchanged)'}", flush=True)
    elif a.cmd == "down":
        cur = current(a.repo, token)
        if not cur or str(cur.get("job")) != str(a.job) or cur.get("status") != "up":
            print(f"[play_link] target.json names job {cur and cur.get('job')} ({cur and cur.get('status')}): left as is",
                  flush=True)
            return
        body = dict(cur, status="down", since=now)
        put_file(a.repo, "target.json", (json.dumps(body, indent=1) + "\n").encode(), f"site down: job {a.job}", token)
        print(f"[play_link] marked down (job {a.job})", flush=True)
    elif a.cmd == "enable-pages":
        st = enable_pages(a.repo, token)
        print(f"[play_link] Pages {'already on' if st == 409 else 'turned on'} for {a.repo}: {pages_url(a.repo)}", flush=True)
    elif a.cmd == "publish-page":
        page = open(PAGE_FILE, encoding="utf-8").read().replace("__REPO__", a.repo)
        changed = put_file(a.repo, "index.html", page.encode(), "forwarding page", token)
        print(f"[play_link] index.html {'published' if changed else 'unchanged'} at {pages_url(a.repo)}", flush=True)


if __name__ == "__main__":
    main()
