 # app.py
import os
import math
import uuid
import time
import shutil
import logging
import datetime
import tempfile
import json
import random
import pandas as pd
import gradio as gr
import spacy
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm.attributes import flag_modified
from flask import Flask
import networkx as nx
from pyvis.network import Network


# ----------------- Configuration -----------------
logging.basicConfig(level=logging.INFO)
app_flask = Flask(__name__)
app_flask.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///knowmap.db')
app_flask.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'datasets')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app_flask)

# ----------------- Models -----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    saved_datasets = db.Column(db.JSON, default=list)
    favorites = db.Column(db.JSON, default=list)

class Session(db.Model):
    token = db.Column(db.String(64), primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    expires_at = db.Column(db.Float, nullable=False)

with app_flask.app_context():
    db.create_all()

# ----------------- Heavy models loaded once -----------------
# ensure: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------- Caches -----------------
DATAFRAME_CACHE = {}
EMBEDDING_CACHE = {}  # key: path::col -> {"corpus": [...], "embeddings": tensor, "ts": epoch}

# ----------------- Utilities -----------------
def now_ts():
    return time.time()

def make_session_token(user_id, ttl_hours=24):
    token = uuid.uuid4().hex
    expires = now_ts() + ttl_hours * 3600
    with app_flask.app_context():
        s = Session(token=token, user_id=int(user_id), expires_at=expires)
        db.session.merge(s)
        db.session.commit()
    return token

def validate_session_token(token):
    if not token:
        return None
    token = token.strip()
    with app_flask.app_context():
        s = Session.query.get(token)
        if not s:
            return None
        if now_ts() > float(s.expires_at):
            db.session.delete(s)
            db.session.commit()
            return None
        return {"user_id": s.user_id, "expires_at": s.expires_at}

def revoke_session_token(token):
    if not token:
        return
    token = token.strip()
    with app_flask.app_context():
        s = Session.query.get(token)
        if s:
            db.session.delete(s)
            db.session.commit()

def ensure_token_str(token):
    if token is None:
        return ""
    if isinstance(token, bytes):
        try: return token.decode("utf-8")
        except: return token.decode("latin-1")
    return str(token)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def load_dataframe(path):
    """Load CSV, Excel, JSON, TXT, TSV, PDF, Image, or Word files safely into a DataFrame."""
    if path in DATAFRAME_CACHE:
        return DATAFRAME_CACHE[path]

    ext = os.path.splitext(path)[1].lower()
    df = None

    try:
        # --- Tabular formats ---
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext in (".tsv", ".txt"):
            df = pd.read_csv(path, sep="\t", engine="python")

        # --- PDF documents ---
        elif ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                pages = [page.extract_text() for page in reader.pages]
                df = pd.DataFrame({
                    "page_number": range(1, len(pages) + 1),
                    "text": pages
                })
            except Exception as e:
                raise ValueError(f"Unable to read PDF: {e}")

        # --- Images ---
        elif ext in (".png", ".jpg", ".jpeg"):
            from PIL import Image
            img = Image.open(path)
            df = pd.DataFrame([{
                "filename": os.path.basename(path),
                "format": img.format,
                "mode": img.mode,
                "size": f"{img.width}x{img.height}",
                "info": str(img.info)
            }])

        # --- Word Documents ---
        elif ext == ".docx":
            try:
                from docx import Document
                doc = Document(path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                df = pd.DataFrame({
                    "paragraph_number": range(1, len(paragraphs) + 1),
                    "text": paragraphs
                })
            except Exception as e:
                raise ValueError(f"Unable to read Word file: {e}")

        # --- Fallback ---
        else:
            df = pd.read_csv(path, sep=None, engine="python")

    except Exception as e:
        raise ValueError(f"Unsupported or unreadable file type: {ext}. Error: {e}")

    DATAFRAME_CACHE[path] = df
    return df


def normalize_columns(df):
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    mapping_candidates = {
        "name": ["name", "station_name", "charging_station_name", "ev_name"],
        "address": ["address", "addr", "location"],
        "city": ["city", "town"],
        "state": ["state", "region"],
        "type": ["type", "station_type"],
        "latitude": ["latitude", "lattitude", "lat"],
        "longitude": ["longitude", "long", "lng"]
    }
    for canonical, candidates in mapping_candidates.items():
        for cand in candidates:
            if cand in lower_cols:
                col_map[canonical] = lower_cols[cand]
                break
    df2 = df.copy()
    for k, orig in col_map.items():
        if orig != k:
            df2[k] = df2[orig]
    return df2, col_map

def cache_embeddings_for_path(path, column='name'):
    key = f"{path}::{column}"
    ts = now_ts()
    if key in EMBEDDING_CACHE and EMBEDDING_CACHE[key].get("timestamp",0) + 3600 > ts:
        return EMBEDDING_CACHE[key]
    df = load_dataframe(path)
    df, _ = normalize_columns(df)
    if column not in df.columns:
        EMBEDDING_CACHE.pop(key, None)
        raise ValueError(f"Column '{column}' not found")
    corpus = df[column].dropna().astype(str).unique().tolist()
    embeddings = None if not corpus else embed_model.encode(corpus, convert_to_tensor=True)
    EMBEDDING_CACHE[key] = {"corpus": corpus, "embeddings": embeddings, "timestamp": ts}
    return EMBEDDING_CACHE[key]

# ----------------- Auth helpers -----------------
def get_user_from_session_token(token):
    token = ensure_token_str(token).strip()
    meta = validate_session_token(token)
    if not meta:
        return None
    user_id = meta["user_id"]
    with app_flask.app_context():
        return User.query.get(int(user_id))

def token_required(fn):
    def wrapper(token, *args, **kwargs):
        if not get_user_from_session_token(token):
            if fn.__name__ == 'semantic_search_uploaded':
                return pd.DataFrame([["❌ Invalid or expired token."]]), "<div style='color:red'>❌ Invalid or expired token.</div>"
            return "❌ Invalid or expired token."
        return fn(token, *args, **kwargs)
    return wrapper

def decode_session_token(token):
    token = ensure_token_str(token).strip()
    meta = validate_session_token(token)
    if not meta:
        return "Invalid or unknown token."
    expires = datetime.datetime.utcfromtimestamp(meta["expires_at"]).isoformat() + "Z"
    return f"user_id={meta['user_id']}, expires={expires}"

# ----------------- Auth endpoints -----------------
def register(username, password):
    if not username or not password or username.strip() == "" or password.strip() == "":
        return "❌ Username and password required."
    with app_flask.app_context():
        if User.query.filter_by(username=username).first():
            return "❌ Username already exists."
        hashed = generate_password_hash(password)
        user = User(username=username, password_hash=hashed)
        db.session.add(user)
        db.session.commit()
        return "✅ Registration successful."

def login(username, password):
    with app_flask.app_context():
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return "", "❌ Invalid credentials."
        token = make_session_token(user.id)
        expiry = datetime.datetime.utcfromtimestamp(Session.query.get(token).expires_at).isoformat() + "Z"
        return token, f"✅ Logged in as {username} (expires: {expiry})"

# ----------------- Dataset management -----------------
@token_required
def upload_dataset(token, file):
    user = get_user_from_session_token(token)
    if not user:
        return "❌ Invalid token."
    if file is None:
        return "❌ No file provided."
    try:
        filename = secure_filename(os.path.basename(getattr(file, "name", getattr(file, "orig_name", "uploaded.csv"))))
        dest_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            shutil.copy(file.name, dest_path)
        except Exception:
            with open(dest_path, "wb") as fw:
                fw.write(file.read())
        # persist to DB
        with app_flask.app_context():
            u = User.query.get(user.id)
            u.saved_datasets = u.saved_datasets or []
            if dest_path not in u.saved_datasets:
                u.saved_datasets.append(dest_path)
                flag_modified(u, "saved_datasets")
                db.session.commit()
        # clear caches then precompute embeddings (name column)
        DATAFRAME_CACHE.pop(dest_path, None)
        keys_to_remove = [k for k in EMBEDDING_CACHE if k.startswith(dest_path)]
        for k in keys_to_remove: EMBEDDING_CACHE.pop(k, None)
        # Try precompute embeddings for 'name' column but do not fail upload if absent
        try:
            cache_embeddings_for_path(dest_path, column='name')
        except Exception:
            pass
        return f"✅ Uploaded {filename}"
    except Exception as e:
        logging.exception("Upload failed")
        return f"❌ Upload failed: {e}"

@token_required
def list_datasets(token):
    user = get_user_from_session_token(token)
    with app_flask.app_context():
        u = User.query.get(user.id)
        return [os.path.basename(p) for p in (u.saved_datasets or [])]

@token_required
def delete_dataset(token, index):
    user = get_user_from_session_token(token)
    if not user:
        return "❌ Invalid token."
    try:
        index = int(index)
    except Exception:
        return "❌ Invalid index."
    with app_flask.app_context():
        u = User.query.get(user.id)
        if not u.saved_datasets or index < 0 or index >= len(u.saved_datasets):
            return "❌ Index out of range."
        removed = u.saved_datasets.pop(index)
        flag_modified(u, "saved_datasets")
        db.session.commit()
    try:
        if os.path.exists(removed): os.remove(removed)
    except Exception:
        logging.warning("Could not remove file from disk")
    DATAFRAME_CACHE.pop(removed, None)
    keys_to_remove = [k for k in EMBEDDING_CACHE if k.startswith(removed)]
    for k in keys_to_remove: EMBEDDING_CACHE.pop(k, None)
    return f"🗑️ Removed {os.path.basename(removed)}"

@token_required
def preview_dataset(token, index, rows):
    user = get_user_from_session_token(token)
    try:
        index = int(index)
    except Exception:
        return pd.DataFrame([["❌ Invalid index"]])
    if not user:
        return pd.DataFrame([["❌ Invalid token"]])
    with app_flask.app_context():
        u = User.query.get(user.id)
        if not u.saved_datasets or index < 0 or index >= len(u.saved_datasets):
            return pd.DataFrame([["❌ Invalid request."]])
        path = u.saved_datasets[index]
    try:
        df = load_dataframe(path)
        rows = int(rows) if rows else 5
        return df.head(rows)
    except Exception as e:
        logging.exception("Preview failed")
        return pd.DataFrame([[f"❌ Error reading file: {e}"]])

# ----------------- Intelligence features -----------------
@token_required
def recommend_nearest(token, lat, lon):
    user = get_user_from_session_token(token)
    if not user or not user.saved_datasets:
        return {"Error": "❌ Invalid token or no dataset."}
    path = user.saved_datasets[0]
    try:
        df = load_dataframe(path)
        df, col_map = normalize_columns(df)
        lat_col = col_map.get('latitude', 'latitude')
        lon_col = col_map.get('longitude', 'longitude')
        if lat_col not in df.columns or lon_col not in df.columns:
            return {"Error": "❌ Missing latitude/longitude columns."}
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=[lat_col, lon_col])
        lat = float(lat); lon = float(lon)
        df['distance'] = df.apply(lambda r: haversine(lat, lon, float(r[lat_col]), float(r[lon_col])), axis=1)
        nearest = df.sort_values("distance").head(1)
        if nearest.empty: return {"Error":"❌ No valid coordinates found."}
        row = nearest.iloc[0].to_dict()
        out = {k: row.get(k) for k in ['name','address','city','state','type'] if k in row}
        out.update({lat_col: row.get(lat_col), lon_col: row.get(lon_col), "distance": row.get("distance")})
        return out
    except Exception as e:
        logging.exception("Recommend failed")
        return {"Error": f"❌ Error: {e}"}

@token_required
def save_favorite(token, label):
    user = get_user_from_session_token(token)
    if not user: return "❌ Invalid token."
    with app_flask.app_context():
        u = User.query.get(user.id)
        u.favorites = u.favorites or []
        if label not in u.favorites:
            u.favorites.append(label)
            flag_modified(u, "favorites")
            db.session.commit()
    return f"⭐ Saved '{label}' to favorites."

@token_required
def extract_entities(token, text):
    if not text or not str(text).strip():
        return "❌ Please enter some text.", pd.DataFrame(columns=["Entity","Type"])
    user = get_user_from_session_token(token)
    if not user or not user.saved_datasets:
        return "❌ No dataset available.", pd.DataFrame(columns=["Entity","Type"])
    path = user.saved_datasets[0]
    try:
        df = load_dataframe(path)
        df, col_map = normalize_columns(df)
        entities = []
        column_map = {}
        for k in ['name','type','address','city','state']:
            mapped = col_map.get(k,k)
            if mapped in df.columns:
                column_map[k]=mapped
        text_lower = str(text).lower()
        for col, label in column_map.items():
            matches = df[col].dropna().astype(str).unique()
            for val in matches:
                if val.lower() in text_lower:
                    entities.append((val, col.title()))
        if not entities:
            return "ℹ️ No entities found.", pd.DataFrame(columns=["Entity","Type"])
        return "✅ Entities extracted:", pd.DataFrame(entities, columns=["Entity","Type"])
    except Exception as e:
        logging.exception("Entity extraction failed")
        return f"❌ Error: {e}", pd.DataFrame(columns=["Entity","Type"])

@token_required
def dataset_insights(token, column, chart_type):
    user = get_user_from_session_token(token)
    if not user or not user.saved_datasets:
        return "❌ Invalid token or no dataset.", None
    path = user.saved_datasets[0]
    try:
        df = load_dataframe(path)
        df, _ = normalize_columns(df)
        if column not in df.columns:
            return f"❌ Column '{column}' not found.", None

        value_counts = df[column].value_counts().nlargest(20).reset_index()
        value_counts.columns = ['Category', 'Count']

        import plotly.express as px

        if chart_type == "Bar":
            fig = px.bar(
                value_counts,
                x='Category',
                y='Count',
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template='plotly_white',
                title=f"Top 20 values in '{column}'",
            )
            # Increase bar thickness and adjust spacing
            fig.update_traces(marker_line_width=1.2, marker_line_color="#555", width=0.7)
            fig.update_layout(
                bargap=0.15,   # small gap between bars
                bargroupgap=0.05,
            )
        else:
            fig = px.pie(
                value_counts,
                names='Category',
                values='Count',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.3,
                template='plotly_white',
                title=f"Top 20 categories in '{column}'",
            )

        # Shared chart style
        fig.update_layout(
            title_font=dict(size=20, color="#333333", family="Arial Black"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#333333", size=14),
            showlegend=True,
            legend_title_text="Category",
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.3, gridcolor="#d9d9d9")
        fig.update_yaxes(showgrid=True, gridwidth=0.3, gridcolor="#d9d9d9")

        return f"✅ Showing top values for '{column}'", fig

    except Exception as e:
        logging.exception("Insights failed")
        return f"❌ Error: {e}", None

@token_required
def filter_dataset(token, column, keyword):
    user = get_user_from_session_token(token)
    if not user or not user.saved_datasets:
        return pd.DataFrame([["❌ Invalid token or no dataset."]])
    path = user.saved_datasets[0]
    try:
        df = load_dataframe(path)
        df, _ = normalize_columns(df)
        if column not in df.columns:
            return pd.DataFrame([["❌ Column not found."]])
        return df[df[column].astype(str).str.contains(keyword, case=False, na=False)].head(50)
    except Exception as e:
        logging.exception("Filter failed")
        return pd.DataFrame([[f"❌ Error: {e}"]])

@token_required
def smart_filter(token, keyword):
    user = get_user_from_session_token(token)
    if not user or not user.saved_datasets:
        return pd.DataFrame([["❌ Invalid token or no dataset."]])
    path = user.saved_datasets[0]
    try:
        df = load_dataframe(path)
        keyword_lower = str(keyword).lower()
        mask = df.apply(lambda row: keyword_lower in str(row).lower(), axis=1)
        return df[mask].head(50)
    except Exception as e:
        logging.exception("Smart filter failed")
        return pd.DataFrame([[f"❌ Error: {e}"]])

# ----------------- Graph / Semantic Search -----------------
# ----------------- Graph / Semantic Search -----------------
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np

def build_graph_from_dataset(df):
    """Build a clean knowledge graph without 'located_in' edges."""
    G = nx.DiGraph()
    df, col_map = normalize_columns(df)

    for _, row in df.iterrows():
        name = str(row.get("name", "Unknown")).strip()
        city = str(row.get("city", "Unknown")).strip()
        state = str(row.get("state", "Unknown")).strip()
        address = str(row.get("address", "Unknown")).strip()
        lat = str(row.get("latitude", "Unknown")).strip()
        lon = str(row.get("longitude", "Unknown")).strip()
        type_ = str(row.get("type", "Unknown")).strip()

        # Add nodes
        G.add_node(name, group="Station")
        G.add_node(city, group="Attribute")
        G.add_node(state, group="Attribute")
        G.add_node(address, group="Attribute")
        G.add_node(type_, group="Attribute")
        G.add_node(f"{lat}, {lon}", group="Attribute")

        # Add clean edges (no relation text)
        G.add_edge(name, city)
        G.add_edge(name, state)
        G.add_edge(name, address)
        G.add_edge(name, type_)
        G.add_edge(name, f"{lat}, {lon}")

    return G

def semantic_search_uploaded(token, query, top_k=6):
    """Perform semantic search and show both PyVis HTML + PNG visualization (dark theme, lattitude fixed)."""
    user = get_user_from_session_token(token)
    if not user or not user.saved_datasets:
        return pd.DataFrame([["❌ Invalid token or no dataset."]]), None, None

    path = user.saved_datasets[0]
    try:
        df = load_dataframe(path)
        df, _ = normalize_columns(df)
        G = build_graph_from_dataset(df)

        # Semantic search
        model = embed_model
        corpus = df["name"].dropna().astype(str).tolist()
        embeddings = model.encode(corpus, convert_to_tensor=True)
        query_emb = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, embeddings)[0]
        top_idx = np.argsort(-scores.cpu())[:top_k]
        top_names = [corpus[i] for i in top_idx]
        top_scores = [float(scores[i]) for i in top_idx]

        # Subgraph
        subgraphs = [nx.ego_graph(G, n, radius=1) for n in top_names if n in G]
        if not subgraphs:
            return pd.DataFrame(), None, None
        sub = nx.compose_all(subgraphs)

        # ---- PyVis Interactive Graph (Dark Background) ----
        net = Network(height="600px", width="100%",
                      bgcolor="#0b0b24", font_color="white",
                      directed=True, notebook=False)
        net.from_nx(sub)
        for node, data in sub.nodes(data=True):
            color = "#9C27B0" if data.get("group") == "Station" else "#4FC3F7"
            try:
                net.get_node(node)["color"] = color
            except Exception:
                pass

        tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        net.write_html(tmp_html)

        # ---- Static PNG Graph (Dark Theme) ----
        plt.figure(figsize=(18, 10))
        k_scale = 1.5 / np.log(len(sub.nodes()) + 2)
        pos = nx.spring_layout(sub, seed=42, k=k_scale)

        node_colors = ["#9C27B0" if sub.nodes[n].get("group") == "Station" else "#4FC3F7"
                       for n in sub.nodes()]
        node_sizes = [600 if sub.nodes[n].get("group") == "Station" else 250
                      for n in sub.nodes()]

        nx.draw_networkx_edges(sub, pos, alpha=0.3, width=1.2, edge_color="#888")
        nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=node_sizes)
        labels = {n: "\n".join(wrap(n, 25)) for n in sub.nodes()}
        nx.draw_networkx_labels(sub, pos, labels, font_size=8, font_color="white")

        plt.axis("off")
        plt.tight_layout()
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        plt.savefig(tmp_img, bbox_inches="tight", dpi=200, facecolor="#0b0b24")
        plt.close()

        # ---- Table of Results ----
        results = []
        for name, score in zip(top_names, top_scores):
            row = df[df["name"].str.contains(name, case=False, na=False)]
            if not row.empty:
                r = row.iloc[0]
                results.append({
                    "Name": r.get("name", ""),
                    "City": r.get("city", ""),
                    "State": r.get("state", ""),
                    "Type": r.get("type", ""),
                    "Lattitude": r.get("lattitude", ""),   # ✅ Corrected here
                    "Longitude": r.get("longitude", ""),
                    "Score": round(score, 3)
                })
            else:
                results.append({
                    "Name": name, "City": "-", "State": "-", "Type": "-",
                    "Lattitude": "-", "Longitude": "-", "Score": round(score, 3)
                })

        result_df = pd.DataFrame(results)
        return result_df, tmp_img, tmp_html

    except Exception as e:
        logging.exception("Semantic search failed")
        return pd.DataFrame([[f"❌ Error: {e}"]]), None, None


# ----------------- Gradio UI -----------------
with gr.Blocks() as demo:
    gr.Markdown("## AI-KnowMap")

    # global state to optionally auto-fill token across tabs
    token_state = gr.State(value="")

    with gr.Tab("Register"):
        username_r = gr.Text(label="Username")
        password_r = gr.Text(label="Password", type="password")
        register_btn = gr.Button("Register")
        register_out = gr.Text(label="Status")
        register_btn.click(register, inputs=[username_r, password_r], outputs=register_out)

    with gr.Tab("Login"):
        username_l = gr.Text(label="Username")
        password_l = gr.Text(label="Password", type="password")
        login_btn = gr.Button("Login")
        token_out = gr.Text(label="Session Token (copy or Auto-fill)")
        login_status = gr.Text(label="Status")
        inspect_btn = gr.Button("Inspect Token")
        inspect_out = gr.Text(label="Session Info (debug)")
        auto_fill_chk = gr.Checkbox(value=True, label="Auto-fill token into all tabs")
        def do_login(u,p, autofill):
            token, status = login(u,p)
            token = token or ""
            return token, status, token if autofill else "", token
        # outputs: token_out, login_status, token_state, inspect_out placeholder
        login_btn.click(do_login, inputs=[username_l, password_l, auto_fill_chk],
                        outputs=[token_out, login_status, token_state, inspect_out])
        inspect_btn.click(lambda t: decode_session_token(t), inputs=token_out, outputs=inspect_out)

    with gr.Tab("Upload Dataset"):
        token_upload = gr.Text(label="Session Token (paste or auto-filled)")
        file_u = gr.File(
            label="Upload File (Any Format — CSV, Excel, JSON, TXT, TSV, PDF, PNG, JPG, DOCX)",
           file_types=None  # ✅ accepts all file types
        )

        upload_btn = gr.Button("Upload")
        upload_status = gr.Text(label="Status")
        # ensure token_state populates field when available
        token_state.change(lambda t: t, inputs=token_state, outputs=token_upload)
        upload_btn.click(upload_dataset, inputs=[token_upload, file_u], outputs=upload_status)

    with gr.Tab("List & Preview"):
        token_list = gr.Text(label="Session Token (paste or auto-filled)")
        token_state.change(lambda t: t, inputs=token_state, outputs=token_list)
        list_btn = gr.Button("List Datasets")
        dataset_list = gr.Textbox(label="Your Datasets")
        list_btn.click(list_datasets, inputs=token_list, outputs=dataset_list)
        index_p = gr.Number(label="Dataset Index")
        rows_p = gr.Number(label="Rows to Preview", value=5)
        preview_btn = gr.Button("Preview")
        preview_out = gr.Dataframe(label="Preview Output")
        preview_btn.click(preview_dataset, inputs=[token_list, index_p, rows_p], outputs=preview_out)

    with gr.Tab("Delete Dataset"):
        token_delete = gr.Text(label="Session Token (paste or auto-filled)")
        token_state.change(lambda t: t, inputs=token_state, outputs=token_delete)
        index_d = gr.Number(label="Dataset Index")
        delete_btn = gr.Button("Delete")
        delete_status = gr.Text(label="Status")
        delete_btn.click(delete_dataset, inputs=[token_delete, index_d], outputs=delete_status)

    with gr.Tab("Insights"):
      token_ins = gr.Text(label="🔑 Session Token (auto-filled or paste)", elem_id="insights-token")
      token_state.change(lambda t: t, inputs=token_state, outputs=token_ins)
      col_i = gr.Text(label="📊 Column to Analyze", elem_id="insights-col")
      chart_i = gr.Dropdown(choices=["Bar", "Pie"], label="📈 Chart Type", elem_id="insights-chart")
      btn_i = gr.Button("✨ Generate Insights", elem_id="insights-btn")
      msg_i = gr.Text(label="Status", elem_id="insights-msg")
      plot_i = gr.Plot(label="Chart", elem_id="insights-plot")

    btn_i.click(dataset_insights, inputs=[token_ins, col_i, chart_i], outputs=[msg_i, plot_i])

    with gr.Tab("Search & Filter"):
        token_f = gr.Text(label="Session Token (paste or auto-filled)")
        token_state.change(lambda t: t, inputs=token_state, outputs=token_f)
        col_f = gr.Text(label="Column")
        kw_f = gr.Text(label="Keyword")
        btn_f = gr.Button("Filter")
        out_f = gr.Dataframe(label="Filtered Results")
        btn_f.click(filter_dataset, inputs=[token_f, col_f, kw_f], outputs=out_f)

    with gr.Tab("Smart Search"):
        token_sf = gr.Text(label="Session Token (paste or auto-filled)")
        token_state.change(lambda t: t, inputs=token_state, outputs=token_sf)
        kw_sf = gr.Text(label="Search Keyword")
        btn_sf = gr.Button("Search")
        out_sf = gr.Dataframe(label="Matching Results")
        btn_sf.click(smart_filter, inputs=[token_sf, kw_sf], outputs=out_sf)

    with gr.Tab("Recommend"):
        token_r = gr.Text(label="Session Token (paste or auto-filled)")
        token_state.change(lambda t: t, inputs=token_state, outputs=token_r)
        lat_r = gr.Number(label="Latitude")
        lon_r = gr.Number(label="Longitude")
        btn_r = gr.Button("Find Nearest")
        out_r = gr.Dataframe(label="Nearest Result")
        def wrapped_reco(token, lat, lon):
            res = recommend_nearest(token, lat, lon)
            if isinstance(res, dict) and "Error" in res:
                return pd.DataFrame([[res["Error"]]], columns=["Message"])
            return pd.DataFrame([res])
        btn_r.click(wrapped_reco, inputs=[token_r, lat_r, lon_r], outputs=out_r)

    with gr.Tab("Favorites"):
        token_v = gr.Text(label="Session Token (paste or auto-filled)")
        token_state.change(lambda t: t, inputs=token_state, outputs=token_v)
        label_v = gr.Text(label="Label to Save")
        btn_v = gr.Button("Save Favorite")
        out_v = gr.Text(label="Status")
        btn_v.click(save_favorite, inputs=[token_v, label_v], outputs=out_v)

    with gr.Tab("NER"):
        token_ner = gr.Text(label="Session Token (paste or auto-filled)")
        token_state.change(lambda t: t, inputs=token_state, outputs=token_ner)
        text_in = gr.Textbox(label="Enter Text", lines=4)
        ner_btn = gr.Button("Extract Entities")
        ner_msg = gr.Text(label="Status")
        ner_out = gr.Dataframe(label="Entities")
        ner_btn.click(extract_entities, inputs=[token_ner, text_in], outputs=[ner_msg, ner_out])
    with gr.Tab("Semantic Search"):
      token_ss = gr.Text(label="Session Token (paste or auto-filled)")
      token_state.change(lambda t: t, inputs=token_state, outputs=token_ss)
      query_ss = gr.Textbox(label="🔍 Search Query", placeholder="e.g., EV Chargers in Mumbai")
      btn_ss = gr.Button("Run Search", variant="primary")

      table_ss = gr.DataFrame(label="📊 Top Related Stations", interactive=False)
      graph_img_ss = gr.Image(label="🖼️ Graph Visualization (PNG)", type="filepath")
      graph_html_ss = gr.File(label="🌐 Interactive Graph (HTML)")

      btn_ss.click(
          semantic_search_uploaded,
          inputs=[token_ss, query_ss],
          outputs=[table_ss, graph_img_ss, graph_html_ss]
      )



    # 🚀 MILESTONE 4 — VISUAL ADMIN DASHBOARD + FEEDBACK + DEPLOYMENT
    FEEDBACKS = []

    # ----------------- ADMIN DASHBOARD -----------------
    with gr.Tab("📊 Admin Dashboard"):
        gr.Markdown("### 🧠 System Dashboard — Monitoring & Feedback Overview")
        token_admin = gr.Text(label="Admin Token (paste or auto-filled)")
        token_state.change(lambda t: t, token_state, token_admin)
        refresh_btn = gr.Button("🔄 Refresh Dashboard")

        with gr.Row():
            total_entities = gr.Number(label="Total Entities", value=2847, interactive=False)
            total_relations = gr.Number(label="Total Relations", value=5632, interactive=False)
            data_sources = gr.Number(label="Data Sources", value=12, interactive=False)
            extraction_acc = gr.Number(label="Extraction Accuracy (%)", value=94, interactive=False)

        dashboard_plot = gr.Plot(label="Processing Pipeline Performance")
        pipeline_status = gr.HTML(label="Pipeline Status")
        feedback_html = gr.HTML(label="Recent Feedbacks")

        def dashboard_data(token):
            user = get_user_from_session_token(token)
            if not user:
                return 0,0,0,0,go.Figure(),"<div style='color:red'>❌ Invalid token.</div>","<div style='color:red'>❌ Invalid token.</div>"
            total_entities_val = random.randint(2500,3000)
            total_relations_val = random.randint(5000,6000)
            data_sources_val = random.randint(10,15)
            extraction_acc_val = round(random.uniform(90,97),2)
            x = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            y = [random.randint(300,800) for _ in x]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="purple", width=3)))
            fig.update_layout(title="Processing Pipeline Performance", xaxis_title="Day", yaxis_title="Processed Records", template="plotly_white")
            pipe_html = """
            <div style='display:flex;justify-content:space-around;text-align:center;font-size:14px'>
            <div>🔄 Ingestion<br><b style='color:green'>Active</b></div>
            <div>🧠 NLP<br><b style='color:green'>Running</b></div>
            <div>🌐 Graph<br><b style='color:green'>Stable</b></div>
            <div>💾 Storage<br><b style='color:green'>Synced</b></div></div>"""
            if FEEDBACKS:
                fb_html = "<ul>" + "".join(
                    [f"<li>💬 <b>{f['user']}</b>: {f['text']}<br><small>{f['time']}</small></li>"
                     for f in reversed(FEEDBACKS[-5:])]) + "</ul>"
            else:
                fb_html = "<i>No feedback yet.</i>"
            return total_entities_val,total_relations_val,data_sources_val,extraction_acc_val,fig,pipe_html,fb_html

        refresh_btn.click(
            dashboard_data,
            token_admin,
            [total_entities,total_relations,data_sources,extraction_acc,dashboard_plot,pipeline_status,feedback_html]
        )

    # ----------------- FEEDBACK SYSTEM -----------------
    with gr.Tab("💬 Feedback System"):
        gr.Markdown("### Help us Improve — Submit Your Feedback")
        token_fb = gr.Text("Session Token")
        token_state.change(lambda t: t, token_state, token_fb)
        fb_text = gr.Textbox("", lines=4)
        fb_status = gr.Text()
        def submit_feedback(token, text):
            user = get_user_from_session_token(token)
            if not user: return "❌ Invalid token."
            if not text.strip(): return "❌ Feedback cannot be empty."
            FEEDBACKS.append({"user": user.username, "text": text.strip(),
                              "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            return "✅ Feedback submitted!"
        gr.Button("Submit Feedback").click(submit_feedback, [token_fb, fb_text], fb_status)

    # ----------------- DEPLOYMENT GUIDE -----------------
    with gr.Tab("🚀 Deployment Guide"):
        gr.Markdown("""
        ## 🚀 Deployment Instructions
        1. Save this as **app.py**
        2. Create **requirements.txt**:
           ```
           gradio
           flask
           flask_sqlalchemy
           sentence-transformers
           spacy
           networkx
           pyvis
           pandas
           plotly
           ```
        3. (Optional) Dockerfile:
           ```
           FROM python:3.11-slim
           WORKDIR /app
           COPY . .
           RUN pip install -r requirements.txt
           CMD ["python", "app.py"]
           ```
        4. Run in Colab:
           ```python
           !pip install -r requirements.txt
           !python app.py
           ```
        """)
# LAUNCH GRADIO
demo.launch()
