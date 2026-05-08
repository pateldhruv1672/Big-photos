"""Small WebHDFS helper layer used by backend, Ray jobs, and scripts.

The project keeps HDFS as the source of truth.  These helpers deliberately avoid
using a database; DataFrames are serialized as Parquet files and binary assets are
written/read through WebHDFS.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib.parse import quote

import pandas as pd
import requests

HDFS_HTTP = os.getenv("HDFS_NAMENODE_HTTP", "http://namenode:9870").rstrip("/")
HDFS_RPC = os.getenv("HDFS_NAMENODE_RPC", "hdfs://namenode:9000").rstrip("/")
HDFS_USER = os.getenv("HDFS_USER", "root")
REQUEST_TIMEOUT = int(os.getenv("HDFS_REQUEST_TIMEOUT", "180"))


def normalize_path(path: str) -> str:
    """Return an absolute HDFS path like /photos/x for hdfs:// or raw paths."""
    if not path:
        raise ValueError("empty HDFS path")
    path = str(path).strip()
    if path.startswith("hdfs://") or path.startswith("webhdfs://"):
        parts = path.split("/", 3)
        path = "/" + parts[3] if len(parts) > 3 else "/"
    if path.startswith(HDFS_RPC):
        path = path[len(HDFS_RPC) :]
    if not path.startswith("/"):
        path = "/" + path
    while "//" in path:
        path = path.replace("//", "/")
    return path


def to_hdfs_uri(path: str) -> str:
    return f"{HDFS_RPC}{normalize_path(path)}"


def parent(path: str) -> str:
    return str(Path(normalize_path(path)).parent)


def _url(path: str, op: str, **params: Any) -> str:
    path = normalize_path(path)
    query = {"op": op, "user.name": HDFS_USER}
    query.update({k: v for k, v in params.items() if v is not None})
    query_string = "&".join(f"{quote(str(k))}={quote(str(v))}" for k, v in query.items())
    return f"{HDFS_HTTP}/webhdfs/v1{quote(path)}?{query_string}"


def mkdirs(path: str) -> bool:
    resp = requests.put(_url(path, "MKDIRS"), timeout=30)
    resp.raise_for_status()
    try:
        return bool(resp.json().get("boolean", False))
    except Exception:
        return True


def exists(path: str) -> bool:
    resp = requests.get(_url(path, "GETFILESTATUS"), timeout=30)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True


def file_status(path: str) -> Optional[Dict[str, Any]]:
    resp = requests.get(_url(path, "GETFILESTATUS"), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("FileStatus")


def delete(path: str, recursive: bool = True) -> bool:
    resp = requests.delete(_url(path, "DELETE", recursive=str(recursive).lower()), timeout=60)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return bool(resp.json().get("boolean", False))


def list_status(path: str) -> List[Dict[str, Any]]:
    resp = requests.get(_url(path, "LISTSTATUS"), timeout=60)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("FileStatuses", {}).get("FileStatus", [])


def walk(path: str) -> Iterator[str]:
    base = normalize_path(path)
    for item in list_status(base):
        child = f"{base.rstrip('/')}/{item['pathSuffix']}"
        if item.get("type") == "DIRECTORY":
            yield from walk(child)
        else:
            yield child


def read_bytes(path: str) -> bytes:
    resp = requests.get(_url(path, "OPEN"), allow_redirects=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def iter_bytes(path: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    with requests.get(_url(path, "OPEN"), allow_redirects=True, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk


def write_bytes(path: str, data: bytes, overwrite: bool = True) -> None:
    path = normalize_path(path)
    mkdirs(parent(path))
    resp = requests.put(
        _url(path, "CREATE", overwrite=str(overwrite).lower()),
        data=data,
        allow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()


def append_bytes(path: str, data: bytes) -> None:
    if not exists(path):
        write_bytes(path, data, overwrite=True)
        return
    resp = requests.post(_url(path, "APPEND"), data=data, allow_redirects=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()


def upload_local_file(local_path: str, hdfs_path: str, overwrite: bool = True) -> None:
    with open(local_path, "rb") as f:
        write_bytes(hdfs_path, f.read(), overwrite=overwrite)


def download_to_local(hdfs_path: str, local_path: str) -> str:
    data = read_bytes(hdfs_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    return local_path


def write_text(path: str, text: str, overwrite: bool = True) -> None:
    write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def read_text(path: str) -> str:
    return read_bytes(path).decode("utf-8")


def write_json(path: str, value: Any, overwrite: bool = True) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str), overwrite=overwrite)


def read_json(path: str, default: Any = None) -> Any:
    try:
        return json.loads(read_text(path))
    except Exception:
        return default


def _download_parquet_files(hdfs_dir: str, local_dir: str) -> List[str]:
    files: List[str] = []
    if not exists(hdfs_dir):
        return files
    for file_path in walk(hdfs_dir):
        if file_path.endswith(".parquet"):
            local_path = os.path.join(local_dir, file_path.strip("/").replace("/", "__"))
            download_to_local(file_path, local_path)
            files.append(local_path)
    return files


def read_parquet_dataset(
    hdfs_dir: str,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Read all Parquet files below an HDFS directory into pandas.

    This is intentionally used only for demo/UI caches and small-to-moderate
    metadata. Batch EDA uses Spark.
    """
    with tempfile.TemporaryDirectory() as tmp:
        files = _download_parquet_files(hdfs_dir, tmp)
        if not files:
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        total = 0
        for file_path in files:
            try:
                df = pd.read_parquet(file_path, columns=columns)
                frames.append(df)
                total += len(df)
                if limit and total >= limit:
                    break
            except Exception as exc:
                print(f"Skipping unreadable Parquet file {file_path}: {exc}")
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True, sort=False)
        return out.head(limit) if limit else out


def write_dataframe_parquet(df: pd.DataFrame, hdfs_dir: str, filename: Optional[str] = None) -> str:
    mkdirs(hdfs_dir)
    if filename is None:
        filename = f"part-{int(time.time())}-{uuid.uuid4().hex[:8]}.parquet"
    local_path = os.path.join(tempfile.gettempdir(), filename)
    df.to_parquet(local_path, index=False)
    hdfs_path = f"{normalize_path(hdfs_dir).rstrip('/')}/{filename}"
    upload_local_file(local_path, hdfs_path, overwrite=True)
    try:
        os.remove(local_path)
    except OSError:
        pass
    return hdfs_path


def pick_existing_path(candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        try:
            if exists(candidate):
                return candidate
        except Exception:
            continue
    return None
