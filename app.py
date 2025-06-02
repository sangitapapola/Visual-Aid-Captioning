import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
import os
from model import CaptionGenerator  # <-- Updated import

# ---------- Configuration ----------
MODEL_PATH = "C:/Majorproject/caption_images_project/model/BestModel1.0"
WORD_TO_INDEX_PATH = "C:/Majorproject/caption_images_project/model/word_to_index.pkl"
INDEX_TO_WORD_PATH = "C:/Majorproject/caption_images_project/model/index_to_word.pkl"

n_heads = 8
num_layers = 3
vocab_size = 8360
embed_size = 512
max_seq_len = 33

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = CaptionGenerator(  # <-- Use renamed class
        embed_size=embed_size,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_heads=n_heads,
        num_layers=num_layers,
        dropout=0.1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_vocab():
    with open(WORD_TO_INDEX_PATH, "rb") as f:
        word_to_index = pickle.load(f)
    with open(INDEX_TO_WORD_PATH, "rb") as f:
        index_to_word = pickle.load(f)
    return word_to_index, index_to_word

@st.cache_data
def get_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ---------- Caption Generation Function ----------
def generate_caption(model, img_tensor, word_to_index, index_to_word, max_seq_len=33, top_k=3):
    device = torch.device("cpu")
    model = model.to(device)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        resnet.eval()
        img_embed = resnet(img_tensor)  # [1, 512, 7, 7]
        img_embed = img_embed.permute(0, 2, 3, 1).reshape(1, -1, 512)  # [1, 49, 512]

    start_token = word_to_index.get("<start>", 1)
    end_token = word_to_index.get("<end>", 2)
    pad_token = word_to_index.get("<pad>", 0)

    input_seq = [pad_token] * max_seq_len
    input_seq[0] = start_token
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)

    predicted_sentence = []
    for i in range(1, max_seq_len):
        output = model(img_embed, input_seq)
        output = output[i - 1, 0, :]
        values, indices = torch.topk(output, top_k)
        next_word_index = indices[0].item()
        next_word = index_to_word.get(next_word_index, "<unk>")
        input_seq[0, i] = next_word_index

        if next_word == "<end>":
            break
        predicted_sentence.append(next_word)

    return " ".join(predicted_sentence)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Visual-Aid Captioning", layout="centered")

# External styling
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">', unsafe_allow_html=True)
if os.path.exists("C:/Majorproject/caption_images_project/static/style.css"):
    with open("C:/Majorproject/caption_images_project/static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
if os.path.exists("C:/Majorproject/caption_images_project/static/script.js"):
    with open("C:/Majorproject/caption_images_project/static/script.js") as f:
        st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="container">
  <h1 class="title">Visual-Aid Captioning</h1>
</div>
""", unsafe_allow_html=True)

# Load model & vocab once
model = load_model()
word_to_index, index_to_word = load_vocab()

# Upload image
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("Generate Caption", key="gen_caption_btn"):
        with st.spinner("Generating caption..."):
            transform = get_transform()
            img_tensor = transform(image)
            caption = generate_caption(model, img_tensor, word_to_index, index_to_word)
            st.success("Caption generated:")
            st.markdown(f"<p id='caption_output' class='caption'>{caption}</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔊 Speak Caption", key="speak_caption"):
                st.markdown(f"""
                <script>
                    const msg = new SpeechSynthesisUtterance("{caption}");
                    window.speechSynthesis.speak(msg);
                </script>
                """, unsafe_allow_html=True)

        with col2:
            if st.button("🗑️ Remove Image", key="remove_image"):
                st.experimental_rerun()
else:
    st.markdown("""
    <div class="drop-area">
      <p>Drag & drop or click to upload an image</p>
    </div>
    """, unsafe_allow_html=True)
