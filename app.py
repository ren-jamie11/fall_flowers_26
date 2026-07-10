import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import html
from collections import Counter

import table

import faulthandler
faulthandler.enable()


DICT_COLUMNS = ["flowers", "leaves", "fruit"]
# Each dict column maps to the plant_type it should restrict results to.
PLANT_TYPE_BY_COLUMN = {"flowers": "flower", "leaves": "leaf", "fruit": "fruit"}
CARD_HEIGHT = 1200 # fixed px height per result card, keeps the 3-col grid aligned

# Columns the UI never reads. Dropped from the cached frame; table.py still
# emits them for the notebooks under Data/.
UNUSED_COLUMNS = ["Price", "Date", "Season", "Category",
                  "colors", "flower_types", "leaf_types", "fruit_types"]

# --- Uniform image sizing ---------------------------------------------------
# Images are never fetched server-side: the browser loads each CDN URL directly
# and crops it to this box with object-fit: cover, so differing source
# dimensions can't break the 3-col grid. Keep the 4:5 ratio or the cards will
# differ in height again.
IMG_ASPECT = "4 / 5"         # 4:5 portrait, suits tall flower-on-stem products
# Where the crop is anchored within the source. "50% 50%" is dead center; lower
# the y (e.g. "50% 40%") to bias toward the top if flower heads get clipped.
IMG_POSITION = "50% 50%"


# --- Data loading -----------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Build the merged/processed DataFrame from table.py (cached across reruns)."""
    df = table.process_table()
    df = df.drop_duplicates(subset=['Product Link', 'Main Img Link'])
    df = df.drop(columns=UNUSED_COLUMNS, errors="ignore")
    # Coerce any pyarrow-backed string columns to plain object dtype. Otherwise
    # value_counts() dispatches to Arrow compute kernels, which segfault on this
    # Python/pyarrow build. object dtype keeps value_counts() on pandas' own code.
    # for c in df.select_dtypes(include="string").columns:
    #     df[c] = df[c].astype(object)
    return df


def _img_html(src: str, alt: str = "⚠️ Could not load image") -> str:
    """Remote image, cover-cropped by the browser — never touches server memory.

    The http->https rewrite avoids mixed-content blocking on the HTTPS-served
    app. It is display-only: table.py joins the CSV's http:// `Main Img Link`
    against the JSON's http:// `img_link`, so rewriting upstream breaks the join.

    No onerror handler — Streamlit strips inline JS. None is needed: the box is
    sized in CSS, so a failed image still fills its slot and shows `alt`.
    """
    src = html.escape(src.replace("http://", "https://", 1), quote=True)
    return (
        f'<img src="{src}" alt="{html.escape(alt)}" loading="lazy" '
        f'style="width:100%;aspect-ratio:{IMG_ASPECT};object-fit:cover;'
        f'object-position:{IMG_POSITION};display:block;border-radius:.5rem;'
        f'background:#eee;color:#666;font-size:.8rem;text-align:center;">'
    )

# --- Dict-column helpers ----------------------------------------------------
def _top_keys(series):
    """Dict keys across a Series, most frequent first."""
    counts = Counter()
    for cell in series:
        if isinstance(cell, dict):
            counts.update(cell.keys())
    return [k for k, _ in counts.most_common()]


def _top_keys_for_colors(series, colors):
    """Keys whose color-set intersects `colors`, most frequent first (one vote
    per row). Mirrors the key-scoped OR filter: a key is counted for a row only
    when that key itself maps to a selected color there — so a keyword offered
    here is one that genuinely has the chosen color(s)."""
    sel = set(colors)
    counts = Counter()
    for cell in series:
        if isinstance(cell, dict):
            for key, cols in cell.items():
                if sel & set(cols):
                    counts[key] += 1
    return [k for k, _ in counts.most_common()]


def _top_colors_for_keys(series, keys):
    """Colors appearing under the selected keys, most frequent first."""
    keyset = set(keys)
    counts = Counter()
    for cell in series:
        if isinstance(cell, dict):
            for key, colors in cell.items():
                if key in keyset:
                    counts.update(colors)
    return [c for c, _ in counts.most_common()]


def _top_colors_flat(series):
    """Colors across a flat `{col}_colors` Series, most frequent first.
    Used for the Color options when no keyword is chosen, so the offered
    colors match exactly what the column-wide OR / Exact Match filters act on."""
    counts = Counter()
    for cell in series:
        counts.update(_color_list_to_set(cell))
    return [c for c, _ in counts.most_common()]


def _row_has_all_keys(cell, keys):
    """True if the dict cell contains every selected key (AND)."""
    return isinstance(cell, dict) and set(keys) <= set(cell.keys())


def _colors_under_keys(cell, keys):
    """Set of colors appearing under the selected keys in a dict cell."""
    if not isinstance(cell, dict):
        return set()
    keyset = set(keys)
    found = set()
    for k, v in cell.items():
        if k in keyset:
            found |= set(v)
    return found


def _row_key_has_color(cell, keys, colors):
    """OR: a selected key maps to at least one selected color."""
    return bool(set(colors) & _colors_under_keys(cell, keys))


def _color_list_to_set(val):
    """Normalize a flat `{col}_colors` cell (list/tuple/set/ndarray) to a set."""
    if isinstance(val, np.ndarray):
        return set(map(str, val.tolist()))
    if isinstance(val, (list, tuple, set)):
        return set(map(str, val))
    return set()


def _row_colors_exactly(color_cell, colors):
    """Exclusive: the row's full color list for this column is EXACTLY the
    selected set, with no extras (e.g. selecting 'brown' keeps a flowers-only
    -brown row but rejects {cream, purple, brown})."""
    return _color_list_to_set(color_cell) == set(colors)


def _nonempty_count(series):
    """Number of rows whose cell is a non-empty dict."""
    return sum(1 for c in series if isinstance(c, dict) and c)


# --- Sorting ----------------------------------------------------------------
INF = float("inf")


def _rank_fn(ranking):
    """value -> its index in `ranking`; unknown values sort last."""
    pos = {v: i for i, v in enumerate(ranking)}
    return lambda v: pos.get(v, INF)


def _best(items, rank):
    """(rank, position) of the best-ranked item; empty sinks to the bottom.
    min() breaks rank ties on the earliest position, so a row leading with the
    winning color outranks one where it appears later."""
    if not items:
        return INF, INF
    i = min(range(len(items)), key=lambda i: rank(items[i]))
    return rank(items[i]), i


def _colors_in_order(cell, keys):
    """Row's colors under `keys` (all keys if none selected), emission order, deduped."""
    if not isinstance(cell, dict):
        return []
    out = []
    for k, colors in cell.items():
        if not keys or k in keys:
            for c in colors:
                if c not in out:
                    out.append(c)
    return out


def sort_results(df, dict_cols, selected_keys):
    """Order rows by product_type, plant_type, then per selected 参数 column:
    species rank, color rank, and the position of that color within the row."""
    if df.empty:
        return df

    keys = pd.DataFrame(index=df.index)

    for col in ("product_type", "plant_type"):
        if col in df.columns:
            rank = _rank_fn(df[col].value_counts().index.tolist())
            keys[col] = df[col].map(rank)

    for col in dict_cols:
        sel = set(selected_keys.get(col) or ())

        species = _rank_fn(_top_keys(df[col]))
        keys[f"{col}_species"] = df[col].map(
            lambda c: _best(list(c) if isinstance(c, dict) else [], species)[0]
        )

        # One row, one vote — and the same lists feed the position below, so the
        # ranking and the position can never disagree.
        rows = df[col].map(lambda c: _colors_in_order(c, sel))
        color = _rank_fn([c for c, _ in
                          Counter(c for lst in rows for c in lst).most_common()])
        keys[f"{col}_color"], keys[f"{col}_pos"] = zip(
            *rows.map(lambda lst: _best(lst, color)))

    if not len(keys.columns):
        return df
    order = keys.reset_index(drop=True).sort_values(
        list(keys.columns), kind="stable").index
    return df.iloc[order]


def _prune_selection(key, options):
    """Auto-deselect any stored value no longer among options (avoids a
    Streamlit error and matches the 'only positive-count options' rule)."""
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]


def filter_dataframe(df: pd.DataFrame, filter_columns=[]) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.
    Widgets are arranged in rows of 3 columns.
    """
    modify = st.checkbox("详细条件", value=True)

    if not modify:
        return df, [], {}

    source = df          # stable reference for building option lists
    df = df.copy()
    selected_keys = {}   # dict column -> keywords chosen for it (drives the sort)

    modification_container = st.container()

    with modification_container:

        # product_type: full-width selector, above the 参数 grid (OR via .isin).
        if "product_type" in source.columns:
            product_types = st.multiselect(
                "product_type",
                options=source["product_type"].value_counts().index.tolist(),
                key="vals_product_type",
            )
            if product_types:
                df = df[df["product_type"].isin(product_types)]

        if not filter_columns:
            filter_columns = df.columns

        # Rank dict columns by non-empty count on the current df; hide empties.
        ranked_cols = sorted(
            (c for c in filter_columns if _nonempty_count(df[c]) > 0),
            key=lambda c: _nonempty_count(df[c]), reverse=True,
        )
        _prune_selection("vals_filter_cols", ranked_cols)
        to_filter_columns = st.multiselect(
            "参数", ranked_cols, key="vals_filter_cols"
        )

        # Arrange dict-column widgets in rows of 3
        for i in range(0, len(to_filter_columns), 3):
            row_cols = st.columns(3)
            for j, column in enumerate(to_filter_columns[i:i + 3]):
                col_widget = row_cols[j]

                # Dict columns: top-N key dropdown + nested color sub-filter.
                col_widget.write(column)

                # Each dict column restricts to its matching plant_type.
                if column in PLANT_TYPE_BY_COLUMN and "plant_type" in df.columns:
                    df = df[df["plant_type"] == PLANT_TYPE_BY_COLUMN[column]]

                # Options must reflect the current selection, but the widgets
                # haven't returned yet — Streamlit already stored the new
                # selections in session_state, so read them from there. This
                # makes Keywords and Color cross-filter each other: the chosen
                # colors narrow the keyword options, just as the chosen keywords
                # narrow the color options below. Fall back to the unnarrowed
                # list when nothing matches, else the dropdown would render empty
                # and strand the user.
                prev = st.session_state.get(f"keys_{column}", [])
                sel_colors = st.session_state.get(f"colors_{column}", [])
                exact = st.session_state.get(f"color_exact_{column}", False)

                # Rows containing every already-chosen keyword (co-occurrence),
                # so multi-key selections stay consistent.
                opt_src = (df[df[column].apply(lambda c: _row_has_all_keys(c, prev))]
                           if prev else df)

                if sel_colors and exact:
                    # Exact Match: keys from rows whose full color list for this
                    # column equals the selection.
                    color_col = f"{column}_colors"
                    exact_rows = opt_src[opt_src[color_col].apply(
                        lambda l: _row_colors_exactly(l, sel_colors))]
                    key_opts = _top_keys(exact_rows[column])
                elif sel_colors:
                    # OR: only keys that themselves map to a chosen color, ranked
                    # by how many rows they appear in.
                    key_opts = _top_keys_for_colors(opt_src[column], sel_colors)
                else:
                    key_opts = _top_keys(opt_src[column])

                key_opts = key_opts or _top_keys(df[column])
                # Never drop a keyword the user already picked just because a
                # color was selected; keep current picks selectable.
                key_opts = list(dict.fromkeys(list(prev) + key_opts))
                _prune_selection(f"keys_{column}", key_opts)
                keys = col_widget.multiselect(
                    "Keywords",
                    options=key_opts,
                    key=f"keys_{column}",
                )
                selected_keys[column] = keys
                st.write(len(key_opts))

                if keys:
                    # AND: keep rows whose dict contains every selected key.
                    df = df[df[column].apply(lambda c: _row_has_all_keys(c, keys))]

                # Color sub-filter renders under Keywords whether or not a
                # keyword is chosen. With keywords, options are scoped to those
                # keys; without, they span every color in the column.
                if keys:
                    color_opts = _top_colors_for_keys(df[column], keys)
                else:
                    color_opts = _top_colors_flat(df[f"{column}_colors"])
                _prune_selection(f"colors_{column}", color_opts)
                colors = col_widget.multiselect(
                    "Color",
                    options=color_opts,
                    key=f"colors_{column}",
                )
                color_and = col_widget.checkbox(
                    "Exact Match", value=False, key=f"color_exact_{column}"
                )
                if colors:
                    color_col = f"{column}_colors"
                    if color_and:
                        # Exclusive match against the row's full color list
                        # for THIS column (e.g. flowers -> flowers_colors).
                        df = df[df[color_col].apply(
                            lambda lst: _row_colors_exactly(lst, colors)
                        )]
                    elif keys:
                        # OR: a selected key maps to a selected color.
                        df = df[df[column].apply(
                            lambda c: _row_key_has_color(c, keys, colors)
                        )]
                    else:
                        # OR, column-wide: the color appears anywhere in this
                        # 参数 for the row (no keyword to scope it to).
                        sel = set(colors)
                        df = df[df[color_col].apply(
                            lambda lst: bool(sel & _color_list_to_set(lst))
                        )]

    return df, to_filter_columns, selected_keys


# --- Streamlit Setup --------------------------------------------------------
st.set_page_config(page_title="Image Keyword Filter", layout="wide")
st.title("🖼️ 人造花图库")

# --- Load DataFrame ---
df = load_data()
filter_columns = ['flowers', 'leaves', 'fruit']

trimmed_df, selected_dict_cols, selected_keys = filter_dataframe(df, filter_columns)

if len(trimmed_df) > 0:
    st.success(f"有{len(trimmed_df)}图片!")
else:
    st.info("No matching images found.")

if st.button("🎨 加载图片"):
    # Sort first, so the cap keeps the top 200 of the hierarchy, not the load order.
    trimmed_sample = sort_results(trimmed_df, selected_dict_cols, selected_keys).head(200)

    def to_str(val):
        """Convert list/ndarray to readable comma-separated string."""
        if isinstance(val, (list, set, tuple)):
            return ", ".join(map(str, val))
        if isinstance(val, np.ndarray):
            return ", ".join(map(str, val.tolist()))
        return str(val)

    def keys_str(cell):
        """Comma-joined dict keys, or '' for empty / non-dict cells."""
        if isinstance(cell, dict) and cell:
            return ", ".join(map(str, cell.keys()))
        return ""

    def colors_str(val):
        """Comma-joined colors list, or '' for empty / NaN cells."""
        if isinstance(val, (list, tuple, set, np.ndarray)):
            return ", ".join(map(str, val))
        return ""

    grid_cols = st.columns(3)

    for idx, (_, row) in enumerate(trimmed_sample.iterrows()):
        with grid_cols[idx % 3]:
            # Fixed-height card keeps every box uniform regardless of field count.
            with st.container(height=CARD_HEIGHT, border=False):
                url = to_str(row.get("Product Link", ""))
                image_url = to_str(row.get("Main Img Link", ""))
                title = to_str(row.get("Name", ""))

                st.html(_img_html(image_url))
                st.caption(title)
                st.caption(url)
                st.caption(image_url)

                lines = [
                    f"**product_type:** {to_str(row.get('product_type', ''))}",
                    f"**plant_type:** {to_str(row.get('plant_type', ''))}",
                ]
                #
                for col in selected_dict_cols:  # only columns chosen as filters
                    ks = keys_str(row.get(col))
                    if ks:  # skip empty dict fields
                        lines.append(f"**{col}:** {ks}")
                    # colors always shown when the field was filtered
                    lines.append(f"**{col}_colors:** {colors_str(row.get(f'{col}_colors'))}")
                lines.append(f"**Store Name:** {to_str(row.get('Store Name', ''))}")
                st.markdown("  \n".join(lines))