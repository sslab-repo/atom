# Installing ATOM

One script, two platforms, everything self-contained in one folder in your
home directory — no system-wide changes, no root on Linux.

```bash
git clone <atom-repo> && cd atom
bash scripts/install.sh
source ~/.bashrc        # RHEL / Linux
source ~/.zshrc         # macOS
atom modules list
```

## What it does

| | RHEL 10 (lab server) | macOS |
|---|---|---|
| Privileges | **user account only — no root, ever** | admin password requested **only** if Python ≥3.10 must be installed via Homebrew |
| Python | system `python3` (RHEL 10 ships 3.12) | existing 3.10+, else `brew install python@3.12` |
| Shell setup | block appended to `~/.bashrc` (bash) | block appended to `~/.zshrc` (zsh) |
| Install folder | `~/atom` | `~/atom` |

Identical layout on both platforms (this is what avoids compatibility
drift):

```
~/atom/
├── app/       copy of the ATOM sources (the install is made from this copy)
├── venv/      private Python virtualenv — never touches system Python
├── bin/atom   launcher (put on PATH); applies your config
├── config/
│   └── atom.env   ALL user configuration lives here (plain env vars)
├── home/      ATOM_HOME: meta-knowledge base, caches
├── data/      your dataset packages (ADPs)
└── runs/      run outputs: provenance + model packages (default --out)
```

The shell block only does two things: sources `config/atom.env` and adds
`~/atom/bin` to PATH. All behavior tuning belongs in `config/atom.env`
(e.g. `ATOM_HOME`, `ATOM_DEFAULT_OUT`) — the file survives upgrades.

## Options

```bash
bash scripts/install.sh --prefix /some/dir   # install elsewhere
bash scripts/install.sh --no-rc              # don't modify shell rc files
bash scripts/install.sh --uninstall          # remove rc block; asks before
                                             # deleting the folder (data!)
```

Re-running the installer upgrades in place: sources are re-copied, the venv
is updated, `config/atom.env` is preserved.

## Self-test

Every install ends with an automatic self-test: `atom modules verify`
(17 module smoke gate) plus a pack → inspect round-trip. If the installer
prints "self-test passed", the installation works.

## Notes

- First install needs network access (PyPI wheels: scikit-learn, pyarrow,
  onnxruntime, skl2onnx).
- The launcher defaults `atom run` output into `~/atom/runs` unless you
  pass `--out`.
- Uninstall never deletes silently: removing `~/atom` (your data, runs,
  meta-KB) requires an explicit yes.
