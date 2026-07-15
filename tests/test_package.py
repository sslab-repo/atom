"""ADP round-trip: pack CSV -> folder ADP -> zip ADP; identical semantics."""

import csv
import shutil
import zipfile

import pytest

from atom.core.ingest import fingerprint
from atom.data import ChecksumMismatch, DatasetPackage, pack_csv

ROWS = 400


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("csvsrc") / "flows.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["duration", " bytes ", "proto", "label"])  # dirty header on purpose
        for i in range(ROWS):
            label = "ATTACK" if i % 10 == 0 else "BENIGN"
            bytes_v = "" if i % 50 == 0 else str(i * 3)  # some missing values
            w.writerow([f"{i}.5", bytes_v, "tcp" if i % 2 else "udp", label])
    return path


@pytest.fixture(scope="module")
def adp_folder(sample_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("packages")
    return pack_csv(sample_csv, out, name="flows-test", target="label")


@pytest.fixture(scope="module")
def adp_zip(adp_folder, tmp_path_factory):
    zdir = tmp_path_factory.mktemp("zips")
    zpath = zdir / "flows-test.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(adp_folder.rglob("*")):
            if f.is_file():
                zf.write(f, f"flows-test/{f.relative_to(adp_folder)}")
    return zpath


def test_pack_produces_valid_manifest(adp_folder):
    with DatasetPackage.open(adp_folder) as pkg:
        m = pkg.manifest
        assert m.manifest_version == "atom-dataset-v1"
        assert m.roles.target == ("label",)
        assert m.labels and m.labels[0]["classes"]["ATTACK"] == ROWS // 10
        assert sum(m.counts[s] for s in ("train", "val", "test")) == ROWS


def test_split_counts_match_parquet(adp_folder):
    with DatasetPackage.open(adp_folder) as pkg:
        for split in ("train", "val", "test"):
            assert pkg.split_row_count(split) == pkg.manifest.counts[split]


def test_typed_processed_columns(adp_folder):
    with DatasetPackage.open(adp_folder) as pkg:
        table = pkg.read_split("train", max_rows=10)
        assert str(table.schema.field("duration").type) == "double"
        assert str(table.schema.field("bytes").type) == "int64"  # header was ' bytes '
        assert str(table.schema.field("proto").type) == "string"


def test_zip_and_folder_identical(adp_folder, adp_zip):
    with DatasetPackage.open(adp_folder) as pf, DatasetPackage.open(adp_zip) as pz:
        assert pf.manifest.content_id == pz.manifest.content_id
        fpf, fpz = fingerprint(pf), fingerprint(pz)
        assert fpf.to_dict() == fpz.to_dict()


def test_fingerprint_contents(adp_folder):
    with DatasetPackage.open(adp_folder) as pkg:
        fp = fingerprint(pkg)
    assert fp.target_classes["BENIGN"] + fp.target_classes["ATTACK"] == fp.sampled_rows
    by_name = {c.name: c for c in fp.columns}
    assert by_name["bytes"].missing_rate > 0
    assert "no-target-declared" not in fp.quality_flags


def test_lazy_checksum_verify_detects_tamper(adp_folder, tmp_path):
    tampered = tmp_path / "tampered"
    shutil.copytree(adp_folder, tampered)
    member = "processed/train.parquet"
    with (tampered / member).open("ab") as fh:
        fh.write(b"x")
    with DatasetPackage.open(tampered) as pkg:
        pkg.verify("processed/val.parquet")  # untouched member passes
        with pytest.raises(ChecksumMismatch):
            pkg.verify(member)


def test_categorical_onehot_expansion(tmp_path):
    import csv as _csv

    from atom.core.dataset import load_matrix
    from atom.core.ingest import fingerprint as _fingerprint

    src = tmp_path / "cats.csv"
    with src.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["num", "color", "label"])
        for i in range(300):
            w.writerow([str(i * 1.5), ["red", "green", "blue"][i % 3], "a" if i % 2 else "b"])
    adp = pack_csv(src, tmp_path, name="cats", target="label")
    with DatasetPackage.open(adp) as pkg:
        fp = _fingerprint(pkg)
        m = load_matrix(pkg, fp, "train", "label")
    assert "num" in m.features
    assert {"color=blue", "color=green", "color=red"} <= set(m.features)
    j = m.features.index("color=red")
    assert set(m.X[:, j]) <= {0.0, 1.0} and m.X[:, j].sum() > 0
