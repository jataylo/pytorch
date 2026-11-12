"""Verify packaged benchmark evidence, optionally checking or extracting raw files."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent


def checked_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe artifact path: {name}")
    return root.joinpath(*relative.parts)


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def verify_file(record):
    path = checked_path(ROOT, record["path"])
    with path.open("rb") as stream:
        actual = digest(stream)
    if actual != record["sha256"]:
        raise RuntimeError(f"Artifact checksum mismatch: {path}")


class JoinedParts(io.RawIOBase):
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def check_archive(archive, expected, destination):
    paths = [checked_path(ROOT, part["path"]) for part in archive["parts"]]
    found = set()
    with io.BufferedReader(JoinedParts(paths)) as stream:
        with tarfile.open(fileobj=stream, mode="r|*") as raw:
            for member in raw:
                if not member.isfile() or member.name not in expected:
                    raise RuntimeError(f"Unexpected archive member: {member.name}")
                if member.name in found:
                    raise RuntimeError(f"Duplicate archive member: {member.name}")
                found.add(member.name)
                record = expected[member.name]
                if member.size != record["bytes"]:
                    raise RuntimeError(f"Raw file size mismatch: {member.name}")
                source = raw.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member: {member.name}")
                value = hashlib.sha256()
                output = None
                if destination is not None:
                    target = checked_path(destination, member.name)
                    if target.is_symlink():
                        raise RuntimeError(f"Refusing symlink target: {target}")
                    if not target.resolve().is_relative_to(destination.resolve()):
                        raise RuntimeError(f"Extraction escapes destination: {target}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = target.open("xb")
                try:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        value.update(block)
                        if output is not None:
                            output.write(block)
                finally:
                    source.close()
                    if output is not None:
                        output.close()
                if value.hexdigest() != record["sha256"]:
                    raise RuntimeError(f"Raw checksum mismatch: {member.name}")
                if destination is not None:
                    target.chmod(member.mode & 0o777)
    if found != set(expected):
        raise RuntimeError(f"Incomplete archive: {archive['name']}")
    print(f"Raw evidence verified: {archive['name']} ({len(found)} files)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deep", action="store_true", help="Verify uncompressed raw files"
    )
    parser.add_argument(
        "--group", action="append", help="Limit verification to an archive group"
    )
    parser.add_argument(
        "--extract", type=Path, help="Extract verified raw files; never overwrite"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "payload/manifest.json").read_text())
    archives = manifest["archives"]
    if args.group:
        unknown = set(args.group) - {archive["name"] for archive in archives}
        if unknown:
            parser.error(f"Unknown archive groups: {sorted(unknown)}")
        archives = [archive for archive in archives if archive["name"] in args.group]
    groups = {archive["name"] for archive in archives}
    verify_file(manifest["raw_manifest"])
    for record in manifest["reports"]:
        verify_file(record)
    for archive in archives:
        for part in archive["parts"]:
            verify_file(part)
    if args.deep or args.extract is not None:
        with gzip.open(
            checked_path(ROOT, manifest["raw_manifest"]["path"]), "rt"
        ) as stream:
            by_group = {name: {} for name in groups}
            for line in stream:
                record = json.loads(line)
                if record["archive"] in groups:
                    by_group[record["archive"]][record["path"]] = record
        for archive in archives:
            check_archive(archive, by_group[archive["name"]], args.extract)
    print(
        f"Verified {len(archives)} archive groups and {len(manifest['reports'])} reports"
    )


if __name__ == "__main__":
    main()
