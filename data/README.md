# Data format

The released code expects a CSV file and a precomputed attribution `.pth` file with one 2-D attribution tensor per CSV row.

Expected CSV columns:

- **IMDb / SST2**: `text,label`
- **BoolQ**: `passage,question,label`
- **TellMeWhy / WikiBio**: `text`

The attribution list must preserve the same sample order as the CSV.

A small IMDb-format example is available in `examples/imdb_sample.csv`.
