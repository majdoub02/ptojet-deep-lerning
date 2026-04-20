
import os, json
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="MedAI Classifier", page_icon="🏥", layout="wide")

@st.cache_resource
def get_class_names():
    for path in ["train"]:
        if os.path.exists(path):
            cls = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))])
            if cls:
                return cls
    return ["NORMAL", "PNEUMONIA"]

@st.cache_resource
def load_models():
    from tensorflow.keras.models import load_model
    loaded = {}
    for name, path in [("VGG16","best_vgg16.keras"),("ResNet50","best_resnet50.keras"),("MobileNetV2","best_mobilenet.keras")]:
        if os.path.exists(path):
            try:
                loaded[name] = load_model(path)
            except:
                pass
    return loaded

def preprocess(img):
    img = img.convert("RGB").resize((224,224))
    return np.expand_dims(np.array(img,dtype=np.float32)/255.0, axis=0)

def predict(model, arr, class_names):
    preds = model.predict(arr, verbose=0)[0]
    top   = int(np.argmax(preds))
    res   = sorted(zip(class_names, preds.tolist()), key=lambda x: x[1], reverse=True)
    return class_names[top], float(preds[top])*100, res

CLASS_NAMES = get_class_names()
MODELS      = load_models()

st.title("🏥 MedAI — Classificateur d'Images Médicales")
st.caption("Deep Learning · Transfer Learning · VGG16 · ResNet50 · MobileNetV2")
st.markdown("---")

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    if MODELS:
        model_choice = st.selectbox("Modèle", list(MODELS.keys()))
    else:
        st.error("❌ Aucun modèle trouvé !")
        model_choice = None
    st.markdown("### Classes détectées")
    for c in CLASS_NAMES:
        st.markdown(f"- {c}")

tab1, tab2, tab3 = st.tabs(["🔬 Classifier", "📊 Performances", "📈 Courbes"])

with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 📤 Image à analyser")
        uploaded = st.file_uploader("Uploader une image", type=["jpg","jpeg","png"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_column_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Largeur", f"{img.size[0]} px")
            c2.metric("Hauteur", f"{img.size[1]} px")

    with col2:
        st.markdown("### 🎯 Résultats")
        if uploaded and MODELS and model_choice:
            with st.spinner(f"Analyse avec {model_choice}..."):
                arr = preprocess(img)
                top_class, top_conf, all_results = predict(MODELS[model_choice], arr, CLASS_NAMES)
            if top_conf >= 75:
                st.success(f"✅ {top_class} — {top_conf:.1f}%")
            elif top_conf >= 50:
                st.warning(f"⚠️ {top_class} — {top_conf:.1f}%")
            else:
                st.error(f"❌ {top_class} — {top_conf:.1f}%")
            st.markdown("**Probabilités :**")
            for cls, prob in all_results:
                st.progress(int(prob*100), text=f"{cls} : {prob*100:.1f}%")
        elif not uploaded:
            st.info("👆 Uploadez une image pour lancer l'analyse.")
        else:
            st.error("❌ Aucun modèle disponible.")

    if uploaded and MODELS and len(MODELS) > 1:
        st.markdown("---")
        st.markdown("### 🏆 Comparaison tous les modèles")
        arr  = preprocess(img)
        cols = st.columns(len(MODELS))
        preds_all = {n: predict(m, arr, CLASS_NAMES) for n,m in MODELS.items()}
        best = max(preds_all, key=lambda x: preds_all[x][1])
        for col, (name, (tc, conf, _)) in zip(cols, preds_all.items()):
            with col:
                st.metric(f"{'🥇 ' if name==best else ''}{name}", tc, f"{conf:.1f}%")

with tab2:
    st.markdown("### 📊 Résultats d'entraînement")
    import pandas as pd
    rows = []
    for name, path in [("VGG16","history_vgg.json"),("ResNet50","history_resnet.json"),("MobileNetV2","history_mobile.json")]:
        if os.path.exists(path):
            with open(path) as f:
                h = json.load(f)
            rows.append({
                "Modèle"    : name,
                "Train Acc" : f"{max(h['accuracy']):.2%}",
                "Val Acc"   : f"{max(h['val_accuracy']):.2%}",
                "Val Loss"  : f"{min(h['val_loss']):.4f}",
                "Epochs"    : len(h['accuracy'])
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Lance d'abord l'entraînement pour voir les résultats.")

with tab3:
    st.markdown("### 📈 Courbes d'apprentissage")
    import pandas as pd
    t1, t2, t3 = st.tabs(["VGG16", "ResNet50", "MobileNetV2"])
    for tab, (name, path) in zip([t1,t2,t3], [("VGG16","history_vgg.json"),("ResNet50","history_resnet.json"),("MobileNetV2","history_mobile.json")]):
        with tab:
            if os.path.exists(path):
                with open(path) as f:
                    h = json.load(f)
                c1, c2 = st.columns(2)
                c1.line_chart(pd.DataFrame({"Train":h["accuracy"],"Validation":h["val_accuracy"]}), height=220)
                c2.line_chart(pd.DataFrame({"Train":h["loss"],"Validation":h["val_loss"]}), height=220)
                st.success(f"✅ Meilleure Val Accuracy : {max(h['val_accuracy']):.2%}")
            else:
                st.info(f"Entraîne {name} pour voir ses courbes.")

st.markdown("---")
st.caption("MedAI · Deep Learning · Transfer Learning · 2025/2026")
