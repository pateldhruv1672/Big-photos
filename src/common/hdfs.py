"""
HDFS utility module — WebHDFS REST client.

Wraps the WebHDFS REST API (port 9870) so every service can read/write
HDFS without needing the Hadoop CLI installed in that container.

Usage:
    from src.common.hdfs import HDFSClient
    client = HDFSClient()
    client.mkdirs("/photos/raw/images")
    client.put("/photos/raw/images/foo.jpg", local_path_or_bytes)
    data = client.get("/photos/raw/images/foo.jpg")
    listing = client.listdir("/photos/raw/images")
"""

import os
import io
import json
import requests
from pathlib import Path
from typing import Union

NAMENODE_HOST = os.environ.get("NAMENODE_HOST", "namenode")
NAMENODE_PORT = int(os.environ.get("NAMENODE_WEBHDFS_PORT", "9870"))
HDFS_USER = os.environ.get("HDFS_USER", "root")
WEBHDFS_BASE = f"http://{NAMENODE_HOST}:{NAMENODE_PORT}/webhdfs/v1"


class HDFSClient:
    """Thin WebHDFS REST client."""

    def __init__(self, host: str = NAMENODE_HOST, port: int = NAMENODE_PORT, user: str = HDFS_USER):
        self.base = f"http://{host}:{port}/webhdfs/v1"
        self.user = user

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _url(self, path: str) -> str:
        """Build a WebHDFS URL for the given HDFS path."""
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base}{path}"

    def _params(self, op: str, **extra) -> dict:
        return {"op": op, "user.name": self.user, **extra}

    # ------------------------------------------------------------------ #
    # Directory operations                                                  #
    # ------------------------------------------------------------------ #
    def mkdirs(self, path: str) -> bool:
        """Create directory (and parents) in HDFS."""
        r = requests.put(self._url(path), params=self._params("MKDIRS"))
        r.raise_for_status()
        return r.json().get("boolean", False)

    def listdir(self, path: str) -> list[dict]:
        """Return list of file status dicts for a directory."""
        r = requests.get(self._url(path), params=self._params("LISTSTATUS"))
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("FileStatuses", {}).get("FileStatus", [])

    def exists(self, path: str) -> bool:
        r = requests.get(self._url(path), params=self._params("GETFILESTATUS"))
        return r.status_code == 200

    # ------------------------------------------------------------------ #
    # File operations                                                       #
    # ------------------------------------------------------------------ #
    def put(self, hdfs_path: str, data: Union[bytes, str, Path], overwrite: bool = True) -> str:
        """
        Upload a file to HDFS.

        Parameters
        ----------
        hdfs_path : str   — destination path in HDFS
        data       : bytes | str | Path — content to upload
        overwrite  : bool — overwrite if exists (default True)

        Returns
        -------
        str — the HDFS URI of the written file
        """
        if isinstance(data, Path):
            data = data.read_bytes()
        elif isinstance(data, str):
            data = data.encode()

        # Step 1: initiate create — get redirect URL
        r = requests.put(
            self._url(hdfs_path),
            params=self._params("CREATE", overwrite=str(overwrite).lower()),
            allow_redirects=False,
        )
        if r.status_code not in (200, 201, 307):
            r.raise_for_status()

        location = r.headers.get("Location")
        if not location:
            # Some versions return 201 directly (no datanode redirect needed in single-node)
            return f"hdfs://{NAMENODE_HOST}:9000{hdfs_path}"

        # Step 2: stream data to datanode
        r2 = requests.put(location, data=data, headers={"Content-Type": "application/octet-stream"})
        r2.raise_for_status()
        return f"hdfs://{NAMENODE_HOST}:9000{hdfs_path}"

    def get(self, hdfs_path: str) -> bytes:
        """Download a file from HDFS and return its raw bytes."""
        r = requests.get(
            self._url(hdfs_path),
            params=self._params("OPEN"),
            allow_redirects=True,
        )
        r.raise_for_status()
        return r.content

    def append(self, hdfs_path: str, data: Union[bytes, str]) -> None:
        """Append bytes to an existing HDFS file."""
        if isinstance(data, str):
            data = data.encode()
        r = requests.post(
            self._url(hdfs_path),
            params=self._params("APPEND"),
            allow_redirects=False,
        )
        location = r.headers.get("Location", "")
        if location:
            r2 = requests.post(location, data=data, headers={"Content-Type": "application/octet-stream"})
            r2.raise_for_status()

    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        r = requests.delete(
            self._url(hdfs_path),
            params=self._params("DELETE", recursive=str(recursive).lower()),
        )
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return r.json().get("boolean", False)

    # ------------------------------------------------------------------ #
    # Convenience                                                           #
    # ------------------------------------------------------------------ #
    def put_local(self, local_path: Union[str, Path], hdfs_dir: str, overwrite: bool = True) -> str:
        """Upload a local file to an HDFS directory, keeping the filename."""
        local_path = Path(local_path)
        hdfs_path = f"{hdfs_dir.rstrip('/')}/{local_path.name}"
        return self.put(hdfs_path, local_path, overwrite=overwrite)

    def uri(self, hdfs_path: str) -> str:
        """Return the hdfs:// URI for a path."""
        return f"hdfs://{NAMENODE_HOST}:9000{hdfs_path}"


# Module-level default client — import and use directly if env vars are set.
default_client = HDFSClient()
