"""Streamlit UI for the flower image classifier (talks to the FastAPI service).

    API_URL=http://localhost:8000 streamlit run app.py
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Flower Image MLOps", page_icon="🌼", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the API; returns (json, error_message)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
    except requests.RequestException as error:
        return None, f"API unreachable at {API_URL}: {error.__class__.__name__}"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"{response.status_code}: {detail}"
    return response.json(), None


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    return " ".join(p for p in (f"{days}d" if days else "", f"{hours}h" if hours else "", f"{minutes}m", f"{secs}s") if p)


st.title("🌼 Flower Image MLOps")
st.caption("Classify flower photos, explore the data, upload new images and retrain the model.")
page = st.sidebar.radio("Navigate", ["Status & up-time", "Predict", "Data visualisations", "Upload data & retrain"])
st.sidebar.caption(f"API: `{API_URL}`")

# --------------------------------------------------------------------------- #
if page == "Status & up-time":
    st.header("Model service status")
    health, error = api("GET", "/health", timeout=5)
    if error:
        st.error(f"🔴 Service DOWN: {error}")
    else:
        if health["status"] == "ok":
            st.success("🟢 Service is up and the model is loaded")
        else:
            st.warning("🟡 Service is up but no model is loaded. Train one first.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Up-time", fmt_duration(health["uptime_seconds"]))
        c2.metric("Started (UTC)", health["started_at"][:19].replace("T", " "))
        c3.metric("Answered by container", health["container"])
        st.caption(f"Model trained at: {health['model_trained_at'] or 'n/a'}")

        stats, _ = api("GET", "/stats")
        if stats:
            usage = stats["usage"]["predictions"]
            st.subheader("Prediction usage")
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Predictions", usage["total"])
            u2.metric("Failed", usage["total"] - usage["succeeded"])
            u3.metric("Mean confidence", f"{usage['mean_confidence']:.1%}" if usage["mean_confidence"] else "n/a")
            u4.metric("Mean latency", f"{usage['mean_duration_ms']:.0f} ms" if usage["mean_duration_ms"] else "n/a")
            if stats["usage"]["class_counts"]:
                st.bar_chart(pd.DataFrame(stats["usage"]["class_counts"]).set_index("name"))
    if st.button("🔄 Refresh"):
        st.rerun()

# --------------------------------------------------------------------------- #
elif page == "Predict":
    st.header("Classify one photo")
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp", "avif"])
    if upload is not None:
        left, right = st.columns([1, 1])
        left.image(upload.getvalue(), caption=upload.name, use_container_width=True)
        if right.button("Predict", type="primary"):
            started = time.perf_counter()
            result, error = api("POST", "/predict", files={"file": (upload.name, upload.getvalue())})
            if error:
                right.error(error)
            else:
                if result["is_flower"]:
                    right.success(f"**{result['label']}** ({result['confidence']:.1%} confident)")
                else:
                    right.warning(f"Not a flower ({result['confidence']:.1%} confident)")
                right.bar_chart(pd.DataFrame(result["top"], columns=["class", "probability"]).set_index("class"))
                right.caption(
                    f"Model latency {result['latency_ms']} ms, round trip "
                    f"{(time.perf_counter() - started) * 1000:.0f} ms, served by {result['container']}"
                )

# --------------------------------------------------------------------------- #
elif page == "Data visualisations":
    st.header("What does the data look like?")
    stats, error = api("GET", "/stats")
    feats, error2 = api("GET", "/features", timeout=120)
    if error or error2:
        st.error(error or error2)
        st.stop()

    st.subheader("1. Class balance")
    balance = pd.DataFrame(stats["dataset"]).fillna(0).astype(int)
    st.bar_chart(balance)
    st.info("Flower classes hold hundreds of images each, but `not_flower` has very few. The model sees "
            "it rarely, so it is the class most likely to be misclassified. Upload more examples to fix this.")

    if not feats["rows"]:
        st.warning("No image statistics available yet. Add training images and retrain.")
        st.stop()
    df = pd.DataFrame(feats["rows"])
    st.caption(f"Feature plots use a random sample of {len(df)} training images.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Colour fingerprint")
        colour = df.groupby("label")[["red", "green", "blue"]].mean()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        colour.plot.bar(ax=ax, color=["#d62828", "#2a9d57", "#1d6fd6"])
        ax.set_ylabel("mean pixel value"); ax.set_xlabel("")
        plt.xticks(rotation=30, ha="right"); st.pyplot(fig)
        st.caption("Sunflowers are blue-poor (yellow); roses are the only class where red clearly exceeds "
                   "green; daisies have the most blue (white petals). Colour is a strong class signal, and "
                   "rose/tulip overlap explains their confusion.")
    with col2:
        st.subheader("3. Brightness")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.boxplot(data=df, x="label", y="brightness", ax=ax, showfliers=False, palette="Set2")
        ax.set_xlabel(""); plt.xticks(rotation=30, ha="right"); st.pyplot(fig)
        st.caption("`not_flower` photos are brighter and far more variable (indoor, products, portraits) "
                   "than outdoor garden shots, so the class is diverse and needs many examples.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("4. Sharpness / texture")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.boxplot(data=df, x="label", y="sharpness", ax=ax, showfliers=False, palette="Set2")
        ax.set_xlabel(""); plt.xticks(rotation=30, ha="right"); st.pyplot(fig)
        st.caption("Daisies have the most edges (small high-contrast petals); dandelions are the softest.")
    with col4:
        st.subheader("5. Image width")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.boxplot(data=df, x="label", y="width", ax=ax, showfliers=False, palette="Set2")
        ax.set_xlabel(""); plt.xticks(rotation=30, ha="right"); st.pyplot(fig)
        st.caption("Flower photos are ~360 px wide, `not_flower` ~830 px: a dataset bias. All images are "
                   "resized to 160x160 so resolution cannot become a shortcut for the model.")

    st.subheader("Model quality (held-out test set)")
    metrics, error = api("GET", "/metrics")
    if error:
        st.warning(error)
    else:
        latest = metrics["latest"]
        m = st.columns(5)
        m[0].metric("Accuracy", f"{latest['accuracy']:.1%}")
        m[1].metric("Macro precision", f"{latest['macro_precision']:.1%}")
        m[2].metric("Macro recall", f"{latest['macro_recall']:.1%}")
        m[3].metric("Macro F1", f"{latest['macro_f1']:.1%}")
        m[4].metric("ROC-AUC", f"{latest['roc_auc_ovr']:.3f}" if latest["roc_auc_ovr"] else "n/a")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(pd.DataFrame(latest["confusion_matrix"], index=latest["classes"], columns=latest["classes"]),
                    annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); st.pyplot(fig)
        per_class = pd.DataFrame(latest["per_class"]).T.drop(index=["macro avg", "weighted avg", "accuracy"], errors="ignore")
        st.dataframe(per_class.round(3), use_container_width=True)

# --------------------------------------------------------------------------- #
else:
    st.header("Upload new data and retrain")
    status, error = api("GET", "/retrain/status")
    if error:
        st.error(error)
        st.stop()

    st.progress(min(status["pending_images"] / status["threshold"], 1.0),
                text=f"{status['pending_images']} new images since last training "
                     f"(auto-retrain at {status['threshold']})")

    st.subheader("1. Upload training data")
    mode = st.radio("Format", ["Images with one label", "ZIP of class folders"], horizontal=True)
    if mode == "Images with one label":
        label = st.text_input("Class label for these images (e.g. daisy, roses, not_flower)")
        files = st.file_uploader("Select multiple images", type=["jpg", "jpeg", "png", "webp", "bmp", "avif"],
                                 accept_multiple_files=True)
    else:
        label = None
        st.code("flowers.zip\n├── daisy/img1.jpg\n├── roses/img2.jpg\n└── not_flower/person.jpg")
        files = st.file_uploader("Select a ZIP", type=["zip"], accept_multiple_files=True)

    if st.button("Upload", disabled=not files):
        payload = [("files", (f.name, f.getvalue())) for f in files]
        result, error = api("POST", "/upload", files=payload, data={"label": label or ""}, timeout=300)
        if error:
            st.error(error)
        else:
            st.success(f"Saved {result['total_saved']} new images: {result['saved']}")
            if result["auto_retrain_started"]:
                st.info("Enough new data arrived: automatic retraining has started.")

    st.subheader("2. Retrain")
    state_text = {"idle": "idle", "running": "⏳ running", "done": "✅ done", "failed": "❌ failed"}
    st.write(f"Status: **{state_text.get(status['state'], status['state'])}**: {status['message']}")
    st.caption(f"Last trained: {status['last_trained_at'] or 'n/a'}")
    if st.button("Retrain model now", type="primary", disabled=status["state"] == "running"):
        _, error = api("POST", "/retrain")
        if error:
            st.error(error)
        else:
            with st.spinner("Retraining..."):
                while True:
                    time.sleep(2)
                    status, _ = api("GET", "/retrain/status")
                    if not status or status["state"] != "running":
                        break
            if status and status["state"] == "done":
                st.success("Retraining complete. The new model is active.")
            else:
                st.error(status["message"] if status else "Lost contact with the API")

    metrics, _ = api("GET", "/metrics")
    if metrics and metrics["history"]:
        st.subheader("Training history")
        history = pd.DataFrame(metrics["history"])
        st.line_chart(history.set_index("trained_at")[["accuracy", "macro_f1"]])
        st.dataframe(history.iloc[::-1], use_container_width=True)
