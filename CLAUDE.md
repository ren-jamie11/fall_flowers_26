# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
streamlit run app.py     # run the app (no build step)
python table.py          # build the table standalone; prints shape/columns
```

There is no test suite, linter, or CI. To sanity-check a change to the data layer, run
`python -c "import table; df = table.process_table(); print(df.shape, list(df.columns))"`
(currently ~4800 rows × 20 columns).

Note: `table.py`'s `__main__` block writes to `Data/Fall Greenery/streamlit_df_output.parquet`, a
directory that does not exist (the real one is `Data/Fall Greenery 2026/`), so running the script
raises at the final line after printing. Nothing consumes that parquet — `app.py` calls
`process_table()` directly.

## Architecture

Two files. `table.py` builds the DataFrame; `app.py` is the Streamlit UI over it.

**This repo is the downstream half of a larger pipeline.** An upstream vision-labeling script
(`ImageTextProcessor.py`, not in this repo) reads `Data/<Season>/Inputs/<store>.csv` — a scraped
product catalog — and emits `Data/<Season>/Outputs/<store>.json` with per-image labels.
`Data/<Season>/Logs/` holds that script's run logs. The notebooks under `Data/` are ad-hoc analysis,
not part of the app.

`table.py:merge_tables` inner-joins the two: every CSV under `Data/` concatenated, joined to every
JSON under `Data/` on `Main Img Link` == `img_link`. Both globs are recursive and unfiltered, so
**any** `.csv`/`.json` added anywhere under `Data/` is silently picked up. Rows with a blank
`plant_type` are then dropped.

### The central data shape

The three label columns — `flowers`, `leaves`, `fruit` — hold dicts of **species → list of colors**:

```python
{"orchid": ["red", "orange"]}
```

Nearly every helper in `app.py` exists to query that shape. `table.py` derives from it:

- `{col}_colors` — flat, deduped color list for one column (`flowers_colors`, etc.)
- `colors` — sorted union across all three
- `{singular}_types` — `flower_types`, `leaf_types`, `fruit_types`: the dict keys

The distinction between a dict column and its `{col}_colors` sibling drives the filter semantics
below: key-scoped filters read the dict, column-wide and Exact Match filters read the flat list.

### Filter semantics (`filter_dataframe`)

Per dict column the user picks Keywords, then Colors, then optionally Exact Match:

- **Keywords: AND** — a row must contain *every* selected key (`_row_has_all_keys`).
- **Colors, with keywords: OR, key-scoped** — some selected key maps to some selected color.
- **Colors, no keywords: OR, column-wide** — the color appears anywhere in that column's flat list.
- **Exact Match** — the row's entire `{col}_colors` list *equals* the selection, no extras. This
  overrides the two OR paths and ignores key scoping.

Selecting a dict column also constrains `plant_type` through `PLANT_TYPE_BY_COLUMN`
(`flowers`→`flower`, `leaves`→`leaf`, `fruit`→`fruit`).

**Keywords and Colors cross-filter each other bidirectionally.** This is the subtlest code in the
repo. Option lists must reflect the current selection, but the widgets haven't returned yet when the
options are computed — so the code reads the already-stored selections out of `st.session_state`
(`keys_{column}`, `colors_{column}`, `color_exact_{column}`) *before* calling `st.multiselect`.
Two consequences to preserve when editing:

- Option lists fall back to the unnarrowed list when narrowing yields nothing, or the dropdown
  renders empty and strands the user.
- `_prune_selection` must drop stored values that are no longer valid options — Streamlit raises
  otherwise. Already-picked keywords are re-inserted into the options for the same reason.

Only options with a positive row count are ever offered, and dict columns with zero non-empty cells
are hidden entirely.

### Sorting and display

`sort_results` builds a composite lexicographic key: `product_type`, `plant_type`, then per selected
column the species rank, color rank, and the position of that color within the row. Ranks come from
frequency within the *current filtered* df, so unknown values sink last (`INF`). Sorting happens
**before** the 200-row cap, so the cap keeps the top of the hierarchy rather than load order.

**Images are never fetched or decoded server-side.** `_img_html` emits an `<img>` pointing straight at
the store's CDN URL, and the browser crops it to `IMG_ASPECT` (4:5) with `object-fit: cover`. Cards
use a fixed `CARD_HEIGHT`. If you change `IMG_ASPECT`, keep it 4:5 or the cards desync.

This is load-bearing for memory, not just style. The app previously cached decoded 800×1000 PIL
images (2.4 MB each) in an unbounded `@st.cache_data`; at 200 images per render against 4310 unique
images, that reached ~10 GB and tripped Streamlit Community Cloud's 690 MB–2.7 GB limit
("This app has gone over its resource limits"). **Do not reintroduce server-side image fetching.**

Two constraints inside `_img_html`:

- The `http://` → `https://` rewrite is **display-only**. It cannot move into `table.py`, because
  `merge_tables` joins the CSV's `http://` `Main Img Link` against the JSON's `http://` `img_link` —
  rewriting either side breaks the inner join.
- No `onerror` handler: `st.html` strips inline JavaScript. None is needed — the box is sized in CSS,
  so a failed image still fills its 4:5 slot and shows the `alt` text over a gray background.

`static.platform.michaels.com` (~200 rows) returns HTTP 403 to everything, browser included, so those
cards show the placeholder. That predates this design and is not fixable client- or server-side.

`load_data` is `@st.cache_data`. Editing `table.py` therefore has no effect until the Streamlit cache
is cleared or the server restarts. It also drops `UNUSED_COLUMNS` — 8 columns the UI never reads;
`table.py` still emits them for the notebooks.

## Conventions

UI strings are partly Chinese (`详细条件` = detailed filters, `参数` = parameters/label columns).
Keep them; they are user-facing labels, not placeholders.

`app.py:321` has a stray `st.write(len(key_opts))` that prints an option count into the page — it
looks like leftover debug output. `app.py` also imports `os` and `random` without using them.
