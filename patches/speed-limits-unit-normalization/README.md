# International units — archived preparation patch

Prepared against `cristianku/sunnypilot:speed-limits@8d37cf946`. After explicit
authorization, the implementation was integrated directly
in `/root/sunnypilot` on `speed-limits`, with tests in that checkout. The archived
patch records the preparation phase; do not reapply it to the already changed
checkout. See `openpilot/sunnypilot/speed_limits/README.md` in the runtime repo.
Borders and GPS policy remain unvalidated; this is not a release ready to deploy.

`runtime.patch` changes vision memory to m/s, adds offline GPS country selection,
preserves old Cap'n Proto field ordinals and rawSignSpeed in physical km/h, and
revokes resolver history when both allowed sources have no current solution.
It also migrates the existing native vision tests to explicit CH context and m/s.
The new regression tests live in `tests/units/` in this repository.

Read-only applicability check **for a clean checkout at the pinned base only**:

```sh
git -C /root/sunnypilot apply --check /root/sunnypilot.speedlimits/patches/speed-limits-unit-normalization/runtime.patch
```

From this repository, `.venv/bin/python -m pytest tests/units -o addopts='' -q`
materializes only the relevant pinned source files in a temporary folder, applies
the patch there and runs the unit/chain/daemon tests plus the 86 migrated native
vision tests. It never creates a git checkout/worktree or modifies sunnypilot.
Set `NNSLR_UNITS_RUNTIME` only to select another local repository with the pinned
commit. No model inference, training, device access, install or download is used.

## Country policy and border evidence

There are **no production defaults** for the GPS policy. `speedvisiond` requires
the release's explicit `settings.country_policy` with all five keys:
`gps_max_age_s`, `gps_max_accuracy_m`, `border_margin_m`, `confirm_s`,
`max_sample_gap_s`. Missing policy or borders leaves the unit unknown and blocks
new numbers; cancellation and physical parking constants still work. GPS loss
holds the physical limit, without introducing a new retention timeout. A
confirmed jurisdiction change clears explicit readings and pending candidates
and re-evaluates implicit defaults; it preserves the physical parking limit.

`country-borders.json.gz` is an **experimental comparison asset**, byte-identical
to `src/nnslr_tools/data/world.json.gz`:

- Natural Earth 1:50m admin-0, public domain; 239 country records.
- Source recorded in the asset:
  `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson`.
- SHA-256: `0bde61b9c68e90f7f780314c22a159612283bf0e48c73be5bcd285009732ec22`.
- It missed the six candidate crossings in the selected real replay windows.
  **It is not suitable for promotion based on these results.**
- The local detailed GISCO file reproduced them, but has non-commercial
  EuroGeographics terms, remains private in the data root, and is not bundled here.

No new borders were downloaded or generated. No map speed, road name, road type
or junction reaches vision. Only country polygons and GPS are used by country.py.
An acceptable border asset and justified policies remain release prerequisites.

See [plan](../../docs/2026-11-02-speed-vision-units-plan.md) and
[verification report](../../docs/2026-11-02-speed-vision-units-report.md).
