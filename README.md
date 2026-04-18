# Inspection DS QC Dashboard

Streamlit dashboard for analyzing inspection DS quality control data.

---

## Folder structure

```
your-repo/
├── app.py                   # Main dashboard
├── convert_to_parquet.py    # One-time local converter
├── requirements.txt
├── README.md
├── .gitignore               # data/ and *.parquet are excluded
└── .streamlit/
    └── secrets.toml         # LOCAL ONLY — never commit this file
```

---

## Step 1 — Convert your CSV to Parquet (run once, on your PC)

Your 700 MB CSV is too large for GitHub (100 MB limit) and too slow
to parse on every app startup. Parquet solves both problems.

```bash
# Install dependencies first
pip install pandas pyarrow

# Run the converter (reads from D:\Insp_Ds\Insp-ds-qc.csv)
python convert_to_parquet.py
```

This creates `data/Insp-ds-qc.parquet` (~60-120 MB, snappy-compressed).

---

## Step 2 — Upload the Parquet file to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Upload `data/Insp-ds-qc.parquet`
3. Right-click the file → **Share** → **Anyone with the link** → **Viewer**
4. Click **Copy link** — it looks like:
   ```
   https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/view
   ```
5. Copy the long ID between `/d/` and `/view`:
   ```
   1aBcDeFgHiJkLmNoPqRsTuVwXyZ
   ```
   That is your **GDRIVE_FILE_ID**.

---

## Step 3 — Test locally

Create `.streamlit/secrets.toml` (this file is git-ignored):

```toml
GDRIVE_FILE_ID = "1aBcDeFgHiJkLmNoPqRsTuVwXyZ"
```

Then run:

```bash
streamlit run app.py
```

The app will find `data/Insp-ds-qc.parquet` directly (no download needed locally).

---

## Step 4 — Push code to GitHub

```bash
git init
git add app.py convert_to_parquet.py requirements.txt README.md .gitignore
git commit -m "Initial dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Verify `.gitignore` is working** — `data/` and `*.parquet` must NOT appear
in `git status`. If they do, run `git rm -r --cached data/` before pushing.

---

## Step 5 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select your repository, branch `main`, and main file `app.py`
4. Click **Advanced settings** → **Secrets** and paste:
   ```toml
   GDRIVE_FILE_ID = "1aBcDeFgHiJkLmNoPqRsTuVwXyZ"
   ```
5. Click **Deploy**

On first load, the app downloads the parquet from Google Drive into `/tmp`
(about 30 seconds). All subsequent interactions use the cached file —
no re-download until the container restarts.

---

## How data loading works

| Situation | What happens |
|---|---|
| Running locally + `data/Insp-ds-qc.parquet` exists | Reads file directly — fastest |
| Deployed + file already in `/tmp` (same session) | Reads from `/tmp` cache — fast |
| Deployed + `/tmp` empty (fresh container) | Downloads from Google Drive once (~30 s) |

---

## Updating the data

When your source CSV changes:
1. Re-run `python convert_to_parquet.py`
2. Re-upload the new `data/Insp-ds-qc.parquet` to Google Drive
   (replace the existing file — the file ID stays the same)
3. In Streamlit Cloud, click **Reboot app** to clear the `/tmp` cache
   so the new file is downloaded on next load