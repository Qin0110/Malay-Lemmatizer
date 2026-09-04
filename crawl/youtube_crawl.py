import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "youtube_raw.jsonl"
STATE_FILE = RAW_DIR / "crawl_state.json"

API = "https://www.googleapis.com/youtube/v3"
DAILY_QUOTA = 10000
MAX_PAGES_PER_VIDEO = 3        
SOURCES = [
    "@safwannazripod",          
    "@thechiccodepodcast",     
    "@legasipodcast",         
    "@kitakitaajepodcast",      
    "@PaduBeb",                
    "@KumaarFamily",          
    "@ichangrm",            
    "@DESTINASITV",
    "@astroawani"
]


class QuotaExceeded(Exception):
    pass


class Client:
    """YouTube Data API wrapper with quota accounting and retries."""

    def __init__(self, key):
        self.key = key
        self.used = 0
        self.last_reason = None       # why the most recent call returned None
        self.session = requests.Session()

    def get(self, endpoint, cost, **params):
        if self.used + cost > DAILY_QUOTA:
            raise QuotaExceeded("would exceed %d units" % DAILY_QUOTA)
        params = {k: v for k, v in params.items() if v is not None}
        params["key"] = self.key
        self.last_reason = None

        for attempt in range(3):
            try:
                r = self.session.get(API + "/" + endpoint, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                print("    network error (%s), retrying" % e)
                time.sleep(2 ** attempt)
                continue

            self.used += cost
            if r.status_code == 200:
                return r.json()

            reason = _reason(r)
            self.last_reason = reason
            if reason == "quotaExceeded":
                raise QuotaExceeded("API reports daily quota spent")
            if reason in ("commentsDisabled", "videoNotFound", "forbidden"):
                return None
            if r.status_code >= 500 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print("    HTTP %s (%s) on %s" % (r.status_code, reason, endpoint))
            return None
        return None


def _reason(response):
    try:
        return response.json()["error"]["errors"][0].get("reason", "unknown")
    except Exception:
        return "unknown"


def uploads_playlist(client, source):
    """Resolve an @handle or channel ID to its uploads playlist."""
    params = {"part": "contentDetails,snippet"}
    params["forHandle" if source.startswith("@") else "id"] = source
    data = client.get("channels", 1, **params)
    if not data or not data.get("items"):
        print("  [skip] cannot resolve %s - check the handle" % source)
        return None, None
    item = data["items"][0]
    return (item["contentDetails"]["relatedPlaylists"]["uploads"],
            item["snippet"]["title"])


def recent_videos(client, playlist_id, limit):
    ids, page = [], None
    while len(ids) < limit:
        data = client.get("playlistItems", 1, part="contentDetails",
                          playlistId=playlist_id,
                          maxResults=min(50, limit - len(ids)), pageToken=page)
        if not data:
            break
        ids += [i["contentDetails"]["videoId"] for i in data.get("items", [])]
        page = data.get("nextPageToken")
        if not page:
            break
    return ids[:limit]


def video_comments(client, video_id, max_pages):
    """Top-level comments plus inline replies. Replies carry the most rojak.

    Returns (rows, reason). reason explains an empty result: "commentsDisabled",
    "empty" (video has no comments yet), or None when rows were returned.
    """
    out, page, reason = [], None, None
    for _ in range(max_pages):
        data = client.get("commentThreads", 1, part="snippet,replies",
                          videoId=video_id, maxResults=100, order="relevance",
                          textFormat="plainText", pageToken=page)
        if not data:
            reason = client.last_reason or "error"
            break
        for thread in data.get("items", []):
            out.append(_row(thread["snippet"]["topLevelComment"], video_id, False))
            for reply in thread.get("replies", {}).get("comments", []):
                out.append(_row(reply, video_id, True))
        page = data.get("nextPageToken")
        if not page:
            break
    if not out and reason is None:
        reason = "empty"
    return out, reason


def _row(comment, video_id, is_reply):
    s = comment["snippet"]
    return {
        "id": comment["id"],
        "video_id": video_id,
        "text": s.get("textOriginal", ""),
        "likes": s.get("likeCount", 0),
        "published_at": s.get("publishedAt", ""),
        "is_reply": is_reply,
    }


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"done_videos": [], "total_comments": 0, "channels": {}}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def read_api_key():
    key = os.environ.get("YOUTUBE_API_KEY")
    if key:
        return key.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("YOUTUBE_API_KEY"):
                return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit(
        "No YOUTUBE_API_KEY found.\n\n"
        "Create a file called .env in the project root containing:\n"
        "    YOUTUBE_API_KEY=your_key_here\n\n"
        "See the docstring at the top of this file for how to get a key."
    )


def crawl(videos_per_channel, max_pages, probe):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(read_api_key())
    state = load_state()
    done = set(state["done_videos"])
    new_comments = 0

    if probe:
        print("PROBE MODE - sampling recent videos, nothing saved.\n")
    else:
        print("Resuming: %d videos done, %d comments on disk\n"
              % (len(done), state["total_comments"]))

    out = None if probe else open(RAW_FILE, "a", encoding="utf-8")
    try:
        for source in SOURCES:
            playlist, title = uploads_playlist(client, source)
            if not playlist:
                continue
            print("%s  (%s)" % (title, source))

            vids = recent_videos(client, playlist, videos_per_channel)
            todo = vids if probe else [v for v in vids if v not in done]
            print("  %d recent videos, %d to crawl" % (len(vids), len(todo)))

            if probe:
                _probe_channel(client, todo, max_pages)
                print()
                continue

            for vid in todo:
                rows, _ = video_comments(client, vid, max_pages)
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                done.add(vid)
                new_comments += len(rows)
                state["done_videos"] = sorted(done)
                state["total_comments"] += len(rows)
                state["channels"][source] = state["channels"].get(source, 0) + len(rows)
                save_state(state)
                print("    %s  +%4d  (quota %d/%d)"
                      % (vid, len(rows), client.used, DAILY_QUOTA))
            print()

    except QuotaExceeded as e:
        print("\nStopped: %s" % e)
        print("Rerun tomorrow with the same command - it resumes automatically.")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved, rerun to resume.")
    finally:
        if out:
            out.close()
            save_state(state)

    print("=" * 60)
    if probe:
        print("Probe done. Swap out any channel whose sample looked too clean,")
        print("then run: python crawl/youtube_crawl.py --videos 25")
    else:
        print("  +%d new comments this run" % new_comments)
        print("  %d total in data/raw/youtube_raw.jsonl" % state["total_comments"])
        print("  %d/%d quota units used" % (client.used, DAILY_QUOTA))
        print("\nNext: python crawl/build_corpus.py")
    print("=" * 60)


def _probe_channel(client, video_ids, max_pages, tries=6):
    skipped = Counter()
    for vid in video_ids[:tries]:
        rows, reason = video_comments(client, vid, max_pages=1)
        if not rows:
            skipped[reason or "empty"] += 1
            continue

        if skipped:
            detail = ", ".join("%d %s" % (n, r) for r, n in skipped.items())
            print("    (skipped %s)" % detail)
        print("    %s: %d comments, 5 samples:" % (vid, len(rows)))
        for r in rows[:5]:
            text = " ".join(r["text"].split())[:88]
            print("      | " + text.encode("ascii", "replace").decode("ascii"))
        return

    detail = ", ".join("%d %s" % (n, r) for r, n in skipped.items()) or "none tried"
    print("    NO COMMENTS in first %d videos (%s)" % (tries, detail))
    print("    -> replace this channel, its uploads are not commentable")


def find_channels(query):
    """Search for channels so you can copy real handles into SOURCES."""
    client = Client(read_api_key())
    data = client.get("search", 100, part="snippet", q=query,
                      type="channel", maxResults=10, regionCode="MY")
    if not data or not data.get("items"):
        print("No channels found for %r" % query)
        return

    ids = [i["snippet"]["channelId"] for i in data["items"]]
    detail = client.get("channels", 1, part="snippet,statistics",
                        id=",".join(ids))
    print("%-28s %-24s %10s" % ("title", "handle", "subs"))
    print("-" * 66)
    for item in (detail or {}).get("items", []):
        s = item["snippet"]
        handle = s.get("customUrl", "(none)")
        subs = item.get("statistics", {}).get("subscriberCount", "?")
        title = s["title"].encode("ascii", "replace").decode("ascii")[:27]
        print("%-28s %-24s %10s" % (title, handle, subs))
    print("\nCopy the handles you want into SOURCES in this file, "
          "then rerun --probe.")
    print("(this search cost %d quota units)" % client.used)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--videos", type=int, default=25,
                   help="recent videos per channel (default 25)")
    p.add_argument("--pages", type=int, default=MAX_PAGES_PER_VIDEO,
                   help="comment pages per video, 100 each (default 3)")
    p.add_argument("--probe", action="store_true",
                   help="sample recent videos per channel, save nothing")
    p.add_argument("--find", metavar="QUERY",
                   help="search for channels by name, print their handles")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    if a.find:
        find_channels(a.find)
    elif a.list:
        print("Channels (edit SOURCES in this file):")
        for s in SOURCES:
            print("  " + s)
    else:
        crawl(a.videos, a.pages, a.probe)
