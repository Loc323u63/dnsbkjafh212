import argparse
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
LYRICS_OVH_URL = "https://api.lyrics.ovh/v1/{artist}/{title}"
LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:120] if cleaned else "untitled"


def fetch_song_titles(artist: str, limit: int = 20, timeout: int = 20):
    params = {
        "term": artist,
        "entity": "song",
        "limit": max(1, min(200, limit * 3)),
    }
    resp = requests.get(ITUNES_SEARCH_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    seen = set()
    titles = []
    for item in data.get("results", []):
        title = (item.get("trackName") or "").strip()
        artist_name = (item.get("artistName") or "").strip()
        if not title or not artist_name:
            continue
        if artist.lower() not in artist_name.lower():
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles




def fetch_lyrics_text_lrclib(artist: str, title: str, timeout: int = 20):
    resp = requests.get(
        LRCLIB_SEARCH_URL,
        params={"artist_name": artist, "track_name": title},
        timeout=timeout,
    )
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
        resp = requests.get(url, timeout=timeout)
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


def download_artist(artist: str, output_dir: Path, max_songs: int, sleep_s: float, dry_run: bool = False):
    titles = fetch_song_titles(artist=artist, limit=max_songs)
    downloaded = []
    skipped = 0

    for title in titles:
        try:
            lyrics = fetch_lyrics_text(artist, title)
            if not lyrics:
                skipped += 1
                continue
            if dry_run:
                downloaded.append(Path(f"{artist}/{title}.txt"))
            else:
                downloaded.append(save_lyrics(output_dir, artist, title, lyrics))
            time.sleep(max(0.0, sleep_s))
        except requests.RequestException:
            skipped += 1
            continue

    return downloaded, skipped, len(titles)


def parse_artists(raw: str):
    return [a.strip() for a in raw.split(",") if a.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", type=str, required=True, help="Исполнители через запятую")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max_songs", type=int, default=20)
    parser.add_argument("--sleep_s", type=float, default=0.25)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    artists = parse_artists(args.artists)
    if not artists:
        raise ValueError("Нужен хотя бы 1 исполнитель")

    total_files = 0
    for artist in artists:
        files, skipped, discovered = download_artist(
            artist=artist,
            output_dir=args.output_dir,
            max_songs=args.max_songs,
            sleep_s=args.sleep_s,
            dry_run=args.dry_run,
        )
        total_files += len(files)
        print(f"Artist: {artist}")
        print(f"  discovered titles: {discovered}")
        print(f"  downloaded: {len(files)}")
        print(f"  skipped/not found: {skipped}")

    print(f"Total downloaded files: {total_files}")


if __name__ == "__main__":
    main()
