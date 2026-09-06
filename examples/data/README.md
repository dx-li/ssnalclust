# Optical Digits data attribution

`optdigits.tes` and `optdigits.names` are unmodified files from the UCI
Optical Recognition of Handwritten Digits archive, retrieved 2026-09-06 UTC.
The library's BSD license does not replace the dataset's license.

**Attribution:** Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of
Handwritten Digits* [Dataset]. UCI Machine Learning Repository.
[DOI: 10.24432/C50P49](https://doi.org/10.24432/C50P49).
The [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
licenses the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original dataset documentation and authorship are preserved in `optdigits.names`.

Download archive:
https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip

SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| Downloaded ZIP | `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0` |
| `optdigits.tes` | `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8` |
| `optdigits.names` | `3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02` |

The file has 1,797 rows, 64 integer features in 0–16 and a final digit-label
column. The study keeps the original row order and all features, divides
features by 16, and withholds labels from fitting and tuning. Its derived
centroid figures and saved arrays arise from this transformation and convex
clustering; they are not unmodified source images. The `.tra` cohort is not
used. No claim of held-out predictive accuracy follows from this study.

## Wine chemical measurements

`wine.data` and `wine.names` are unchanged files from the UCI Wine archive:
https://archive.ics.uci.edu/static/public/109/wine.zip

Attribution: Aeberhard, S. and Forina, M. (1992). *Wine* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5PC7J. The UCI dataset page
https://archive.ics.uci.edu/dataset/109/wine lists CC BY 4.0 licensing.
The included `wine.names` preserves the original accompanying metadata.

The data have 178 rows, 13 chemical features, and a first-column cultivar label
(1–3). The label is separated before preprocessing and is used only for
posthoc evaluation. Training-only scaling is a transformation performed by
the held-out-entry study; these source files are unchanged.

SHA256:

- ZIP: `2bae62c4481220623579d4c4fb36b55652b6b75e06e49fa1981b8198362dfdab`
- `wine.data`: `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659`
- `wine.names`: `f1b84f2ef845e0bdebf13e14fa7a213e56de4f1baa40c5974dbd1ee51c5ae710`
