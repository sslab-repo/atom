"""M6 groundwork: image ADP packaging + fingerprint + graceful run gate."""

import pytest

from atom.core.ingest import fingerprint
from atom.core.run import run_package
from atom.core.task_inference import infer
from atom.data import DatasetPackage, pack_images

PNG_STUB = bytes.fromhex("89504e470d0a1a0a") + b"atomtest" * 4  # not decodable; v0 profiles metadata only


@pytest.fixture(scope="module")
def image_adp(tmp_path_factory):
    src = tmp_path_factory.mktemp("imgs")
    for cls in ("cat", "dog"):
        d = src / cls
        d.mkdir()
        for i in range(8):
            (d / f"{cls}_{i}.png").write_bytes(PNG_STUB + bytes([i]))
    return pack_images(src, tmp_path_factory.mktemp("pkg"), name="pets")


def test_image_adp_fingerprint(image_adp):
    with DatasetPackage.open(image_adp) as pkg:
        assert pkg.manifest.mode == "files"
        assert pkg.manifest.modality == "image"
        fp = fingerprint(pkg)
    assert sum(fp.counts[s] for s in ("train", "val", "test")) == 16
    assert set(fp.target_classes) == {"cat", "dog"}
    spec = infer(fp)
    assert spec.family.value == "classification"
    assert spec.modality.value == "image"


def test_image_run_gated_gracefully(image_adp, tmp_path):
    with pytest.raises(SystemExit, match="foundation adapters"):
        run_package(str(image_adp), wall_clock_s=10, out_root=str(tmp_path))


def test_image_checksums_verify(image_adp):
    with DatasetPackage.open(image_adp) as pkg:
        member = pkg.manifest.files[0]["path"] if isinstance(pkg.manifest.files[0], dict) \
            else pkg.manifest.files[0].path
        pkg.verify(member)
        pkg.verify("processed/train.jsonl")
