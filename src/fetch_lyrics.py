import argparse
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
YOUTUBE_SEARCH_URL = "https://www.youtube.com/results"
SOUNDCLOUD_SEARCH_URL = "https://soundcloud.com/search/sounds"
VK_SEARCH_URL = "https://m.vk.com/search"
LYRICS_OVH_URL = "https://api.lyrics.ovh/v1/{artist}/{title}"
LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"

DEFAULT_SOURCES = ["itunes", "youtube", "soundcloud", "vk"]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(REQUEST_HEADERS)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:120] if cleaned else "untitled"


def normalize_title(title: str) -> str:
    title = re.sub(r"\s*\(.*?(official|lyrics?|audio|video).*?\)\s*", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\[.*?(official|lyrics?|audio|video).*?\]\s*", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+-\s+.*?(official|lyrics?|audio|video).*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" -")
    return title


def fetch_song_titles_itunes(artist: str, limit: int = 20, timeout: int = 20):
    params = {"term": artist, "entity": "song", "limit": max(1, min(200, limit * 3))}
    resp = SESSION.get(ITUNES_SEARCH_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    titles = []
    for item in data.get("results", []):
        title = (item.get("trackName") or "").strip()
        artist_name = (item.get("artistName") or "").strip()
        if title and artist_name and artist.lower() in artist_name.lower():
            titles.append(title)
    return titles


def fetch_song_titles_youtube(artist: str, limit: int = 20, timeout: int = 20):
    query = f"{artist} official audio"
    resp = SESSION.get(YOUTUBE_SEARCH_URL, params={"search_query": query}, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    matches = re.findall(r'"title":\{"runs":\[\{"text":"([^"]{2,120})"\}\]\}', html)
    titles = []
    for raw in matches:
        candidate = normalize_title(raw)
        if not candidate:
            continue
        if artist.lower() not in candidate.lower() and len(candidate.split()) < 2:
            continue
        titles.append(candidate)
        if len(titles) >= limit * 3:
            break
    return titles


def fetch_song_titles_soundcloud(artist: str, limit: int = 20, timeout: int = 20):
    resp = SESSION.get(SOUNDCLOUD_SEARCH_URL, params={"q": artist}, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    matches = re.findall(r'href="/[^"/]+/([^"?#]+)"', html)
    titles = []
    for slug in matches:
        candidate = normalize_title(slug.replace("-", " "))
        if candidate and len(candidate) >= 3:
            titles.append(candidate)
        if len(titles) >= limit * 3:
            break
    return titles


def fetch_song_titles_vk(artist: str, limit: int = 20, timeout: int = 20):
    resp = SESSION.get(VK_SEARCH_URL, params={"c[q]": f"{artist}", "c[section]": "audio"}, timeout=timeout)
    if resp.status_code >= 400:
        return []
    html = resp.text
    matches = re.findall(r'class="audio_item__title[^>]*>\s*([^<]{2,140})\s*<', html)

    titles = []
    for m in matches:
        candidate = normalize_title(m)
        if candidate:
            titles.append(candidate)
        if len(titles) >= limit * 3:
            break
    return titles


def dedupe_titles(titles, artist: str, limit: int):
    seen = set()
    out = []
    for title in titles:
        key = title.lower().strip()
        if not key or key in seen:
            continue
        if key == artist.lower().strip():
            continue
        seen.add(key)
        out.append(title)
        if len(out) >= limit:
            break
    return out


def fetch_song_titles(artist: str, limit: int = 20, timeout: int = 20, sources=None):
    sources = sources or DEFAULT_SOURCES
    all_titles = []

    for source in sources:
        try:
            if source == "itunes":
                all_titles.extend(fetch_song_titles_itunes(artist, limit=limit, timeout=timeout))
            elif source == "youtube":
                all_titles.extend(fetch_song_titles_youtube(artist, limit=limit, timeout=timeout))
            elif source == "soundcloud":
                all_titles.extend(fetch_song_titles_soundcloud(artist, limit=limit, timeout=timeout))
            elif source == "vk":
                all_titles.extend(fetch_song_titles_vk(artist, limit=limit, timeout=timeout))
        except requests.RequestException:
            continue

    return dedupe_titles(all_titles, artist=artist, limit=limit)


def fetch_lyrics_text_lrclib(artist: str, title: str, timeout: int = 20):
    resp = SESSION.get(LRCLIB_SEARCH_URL, params={"artist_name": artist, "track_name": title}, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    results = resp.json()
    if not isinstance(results, list):
        return None
    for item in results:
        cand = (item.get("plainLyrics") or item.get("syncedLyrics") or "").strip()
        if len(cand) >= 60:
            return cand
    return None


def fetch_lyrics_text(artist: str, title: str, timeout: int = 20):
    url = LYRICS_OVH_URL.format(artist=quote(artist), title=quote(title))
    try:
        resp = SESSION.get(url, timeout=timeout)
        if resp.status_code != 404:
            resp.raise_for_status()
            data = resp.json()
            lyrics = (data.get("lyrics") or "").strip()
            if len(lyrics) >= 60:
                return lyrics
    except requests.RequestException:
        pass

    try:
        return fetch_lyrics_text_lrclib(artist=artist, title=title, timeout=timeout)
    except requests.RequestException:
        return None


def save_lyrics(base_dir: Path, artist: str, title: str, lyrics: str):
    artist_dir = base_dir / sanitize_filename(artist)
    artist_dir.mkdir(parents=True, exist_ok=True)
    path = artist_dir / f"{sanitize_filename(title)}.txt"
    path.write_text(lyrics.strip() + "\n", encoding="utf-8")
    return path


def download_artist(artist: str, output_dir: Path, max_songs: int, sleep_s: float, dry_run: bool = False, sources=None):
    titles = fetch_song_titles(artist=artist, limit=max_songs, sources=sources)
    downloaded = []
    skipped = 0

    for title in titles:
        lyrics = fetch_lyrics_text(artist, title)
        if not lyrics:
            skipped += 1
            continue
        if dry_run:
            downloaded.append(Path(f"{artist}/{title}.txt"))
        else:
            downloaded.append(save_lyrics(output_dir, artist, title, lyrics))
        time.sleep(max(0.0, sleep_s))

    return downloaded, skipped, len(titles)


def parse_artists(raw: str):
    return [a.strip() for a in raw.split(",") if a.strip()]


def parse_sources(raw: str):
    allowed = set(DEFAULT_SOURCES)
    sources = [s.strip().lower() for s in raw.split(",") if s.strip()]
    sources = [s for s in sources if s in allowed]
    return sources or ["itunes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", type=str, required=True, help="Исполнители через запятую")
    parser.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES), help="Источники: itunes,youtube,soundcloud,vk")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max_songs", type=int, default=20)
    parser.add_argument("--sleep_s", type=float, default=0.25)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    artists = parse_artists(args.artists)
    if not artists:
        raise ValueError("Нужен хотя бы 1 исполнитель")
    sources = parse_sources(args.sources)

    total_files = 0
    for artist in artists:
        files, skipped, discovered = download_artist(
            artist=artist,
            output_dir=args.output_dir,
            max_songs=args.max_songs,
            sleep_s=args.sleep_s,
            dry_run=args.dry_run,
            sources=sources,
        )
        total_files += len(files)
        print(f"Artist: {artist}")
        print(f"  sources: {', '.join(sources)}")
        print(f"  discovered titles: {discovered}")
        print(f"  downloaded: {len(files)}")
        print(f"  skipped/not found: {skipped}")

    print(f"Total downloaded files: {total_files}")


if __name__ == "__main__":
    main()
