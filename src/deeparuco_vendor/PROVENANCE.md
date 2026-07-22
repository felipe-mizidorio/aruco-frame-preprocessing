# Provenance — `src/deeparuco_vendor/`

This directory is a vendored copy of third-party code. It is **not** authored by this
project. This note records where it came from, which version, when it was brought in, and
how it has diverged since, so that the DeepArUco++ comparison produced by
`src/deeparuco_comparison.py` can be re-derived by a third party or by this project after
the upstream has moved on.

## Upstream

| Item | Value |
|---|---|
| Project | DeepArUco++ |
| Repository | <https://github.com/AVAuco/deeparuco> |
| Source directory | `impl/` |
| Upstream ref | tag **`IMAVIS`** |
| Upstream commit | `03f49822be191f04fadd479008a02187be5844ea` |
| Files taken | `aruco.py`, `heatmaps.py`, `losses.py`, `utils.py` |
| Vendored on | 2026-06-01 |
| Vendored by commit | `89c12b2` — *"feat: vendor deeparuco impl/ from AVAuco/deeparuco"* |

`__init__.py` is a local addition (empty) to make the directory a package. It has no
upstream counterpart.

### How the version was established

The vendoring commit records the project but not the upstream revision, so the version was
recovered by content comparison rather than taken on trust:

- `impl/` upstream contains exactly eight modules; the four vendored here are a subset.
- Upstream has two tags, `IbPRIA'23` and `IMAVIS`. At `IbPRIA'23` (`b8ffdc9`) the `impl/`
  directory does not exist, so that tag is excluded.
- At tag `IMAVIS` (`03f4982`) all four files are present, and their contents match the
  files vendored here (see *Divergence* below for the exact sense of "match").
- `impl/` has not been modified upstream since commit `16695f3` (2024-11-08). The tag
  `IMAVIS` and the current default branch (`main`) therefore carry identical `impl/`
  content, and the identification does not depend on which of the two is used as the
  reference.

Upstream has no packaged release or version number; the tag and commit SHA above are the
most precise identifiers available.

## Divergence from upstream

Two separate kinds of change apply. They are recorded separately because only the second
affects behaviour.

### 1. Reformatting, applied when the files were copied in (commit `89c12b2`)

The files were passed through the project's formatter (black + isort) as they were
vendored. Verified by parsing both sides with `ast` and comparing the trees:

| File | Byte-identical to upstream | Semantically identical to upstream |
|---|---|---|
| `heatmaps.py` | **yes** | yes |
| `losses.py` | no | yes — line wrapping only |
| `aruco.py` | no | yes — apart from import order (isort) |
| `utils.py` | no | yes — apart from import order (isort) |

`heatmaps.py` is byte-for-byte identical to upstream, which anchors the identification
independently of any formatting argument. For the other three the abstract syntax trees are
equal except for the order of top-level imports, which isort rewrites and which does not
change behaviour here (no import has a side effect that another depends on).

SHA-256 of the four files **as vendored** at `89c12b2`:

```
aruco.py     5ccfae925032871257454788e55d3e26406efb4a37246f8aa88a0e7756e46372
heatmaps.py  fc2ca5d1d03989a21edfe955ce0dea3e6334a8ac1fff3500333e388be357982f
losses.py    7326a34693b4fc151da3ee8e008c67e0602253f95a9ad2a0458e1c5b722879f7
utils.py     8c966c1213a31b65e2f9d8e9726bc4be7891dd1aa4b1c7a674142fa7b3bee2ec
```

SHA-256 of upstream `impl/` at `03f4982`, for comparison:

```
aruco.py     6cb9cebfb2f852bc3c41ab1e6771d2b154f6a1b77c6b7be7e0ce1ff4c1634779
heatmaps.py  fc2ca5d1d03989a21edfe955ce0dea3e6334a8ac1fff3500333e388be357982f
losses.py    077619d5c0307ad9b4cfac33dc8610618ad3bc5682a59ecc4e5a89e9cd3b2926
utils.py     b16e14309a48a03677a4192ad5b8e281bfbc278b33a173d1cbc6c629dd5559c6
```

### 2. One deliberate behavioural fix, made after vendoring (commit `5806b2a`, 2026-06-03)

`utils.py`, function `ordered_corners` — the arguments to `np.arctan2` were swapped:

```diff
-    angles = np.arctan2(x_vals - cx, y_vals - cy)
+    angles = np.arctan2(y_vals - cy, x_vals - cx)
```

`np.arctan2` takes `(y, x)`. The upstream call passes `(x, y)`, which reflects the computed
angle about the 45° line and so produces a different corner ordering; that ordering feeds
the perspective warp applied to every marker crop before decoding.

**This is the one point where the vendored code no longer matches upstream behaviour.** Any
comparison run from this tree is a comparison against *patched* DeepArUco++, not stock
DeepArUco++, and should be described that way. The change has not been offered upstream.

No other file has been modified since vendoring. `aruco.py`, `heatmaps.py` and `losses.py`
are unchanged from the state recorded in the first hash table above.

## Verifying this note

From a checkout of this repository, with network access:

```bash
# 1. Fetch upstream impl/ at the identified commit
for f in aruco.py heatmaps.py losses.py utils.py; do
  curl -sO "https://raw.githubusercontent.com/AVAuco/deeparuco/03f49822be191f04fadd479008a02187be5844ea/impl/$f"
done

# 2. heatmaps.py should be byte-identical to the vendored copy
sha256sum heatmaps.py src/deeparuco_vendor/heatmaps.py

# 3. The rest should differ only by formatting, and utils.py additionally by the
#    arctan2 fix documented above
python -c "import ast,sys; f=lambda p: ast.dump(ast.parse(open(p,encoding='utf-8').read())); print(f(sys.argv[1])==f(sys.argv[2]))" \
  aruco.py src/deeparuco_vendor/aruco.py
```

## Maintenance

- **Do not edit these files** for style or lint reasons. They are excluded from ruff and
  pyright in `pyproject.toml` precisely so that they can stay close to upstream.
- If a behavioural change becomes necessary, add it to the *Divergence* section above with
  the commit SHA and the reason. A divergence that is not recorded here makes the
  comparison unreproducible again.
- If the files are re-synced from a newer upstream, update the tag, commit SHA, date and
  both hash tables, and re-check whether the `arctan2` fix is still needed or has been
  fixed upstream.

## Model weights

The comparison also depends on three model weight files that are **not** vendored.
`deeparuco_comparison.py` downloads them on first run from
`https://raw.githubusercontent.com/AVAuco/deeparuco/master/models` into
`~/.cache/deeparuco/`.

That URL is pinned to a **branch, not a commit**, so it tracks whatever the upstream branch
points at rather than a fixed revision, and the downloaded files are not checksummed. The
weights are therefore the remaining unpinned input to the comparison: this note makes the
vendored *code* reproducible, but a future download could differ from the one used to
produce any given `comparison.json` without that being detectable. Recording the weight
hashes at run time, or pinning the URL to a commit SHA, would close that gap — both are
behavioural changes and are deliberately left out of scope here.
