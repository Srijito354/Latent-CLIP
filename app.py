import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ----------------- Utility -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

embedding_dim = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_embed_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(16, 16), stride=16).to(device)

# ----------------- Embedding & Positional Encoding -----------------
def embedding_gen(sentence):
    global words, word_idx, vocab_size, idx_only
    words = sentence.lower().split()
    word_idx = {word: idx for idx, word in enumerate(words)}
    idx_only = list(range(len(word_idx)))
    vocab_size = len(word_idx)
    embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    input_tensor = torch.LongTensor(idx_only)
    input_embeddings = embeddings(input_tensor)
    return input_embeddings

def positional_encodings(sequence_length, embedding_size):
    pe = torch.zeros(sequence_length, embedding_size)
    for pos in range(sequence_length):
        for i in range(embedding_size):
            if i % 2 == 0:
                pe[pos, i] = torch.sin(torch.tensor(pos / (10000 ** ((2 * i) / embedding_size))))
            else:
                pe[pos, i] = torch.cos(torch.tensor(pos / (10000 ** ((2 * i) / embedding_size))))
    return pe

# ----------------- Model Definitions -----------------
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.embedding_dim = 16
        self.w_q = nn.Linear(16, 2)
        self.w_k = nn.Linear(16, 2)
        self.w_v = nn.Linear(16, 2)
        self.latent_upscale = nn.Linear(2, 16)
        self.layer_norm = nn.LayerNorm(16)
        self.feed_fwd = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16)
        )
        self.output_proj = nn.Linear(16, 16)

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        context = self.latent_upscale(context)
        x = self.layer_norm(context + x)
        ff_out = self.feed_fwd(x)
        out = self.layer_norm(ff_out + x)
        return self.output_proj(out)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.embedding_dim = 16
        self.w_q = nn.Linear(16, 2)
        self.w_k = nn.Linear(16, 2)
        self.w_v = nn.Linear(16, 2)
        self.latent_upscale = nn.Linear(2, 16)
        self.layer_norm = nn.LayerNorm(16)
        self.feed_fwd = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16)
        )
        self.output_proj = nn.Linear(16, 16)

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_k(x)  # Intentional duplication
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        context = self.latent_upscale(context)
        x = self.layer_norm(context + x)
        ff_out = self.feed_fwd(x)
        out = self.layer_norm(ff_out + x)
        return self.output_proj(out)

class CLIPMini(nn.Module):
    def __init__(self):
        super(CLIPMini, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, image_patches, text_tokens):
        cls_token_img = nn.Parameter(torch.randn(1, 1, 16)).to(image_patches)
        cls_token_img = cls_token_img.expand(image_patches.size(0), -1, -1)
        img_input = torch.cat([cls_token_img, image_patches], dim=1)
        img_embs = self.image_encoder(img_input)

        cls_token_txt = nn.Parameter(torch.randn(1, 1, 16)).to(text_tokens.device)
        cls_token_txt = cls_token_txt.expand(text_tokens.size(0), -1, -1)
        txt_input = torch.cat([cls_token_txt, text_tokens], dim=1)
        txt_embs = self.text_encoder(txt_input)

        img_vec = F.normalize(img_embs[:, 0, :], dim=-1)
        txt_vec = F.normalize(txt_embs[:, 0, :], dim=-1)
        return img_vec, txt_vec

# ----------------- Load Model -----------------
model = CLIPMini()
model.load_state_dict(torch.load("clip_mini2.pth", map_location=device))
model.to(device)
model.eval()

# ----------------- Inference -----------------
def run_inference(image, question):
    img = transform(image).unsqueeze(0).to(device)
    img_patches = img_embed_layer(img).flatten(2).transpose(1, 2)
    pos_mat = torch.randn(1, 196, 16).to(device)
    img_input = img_patches + pos_mat

    q_emb = embedding_gen(question)
    q_pe = positional_encodings(len(word_idx), embedding_dim)
    question_input = (q_emb + q_pe).unsqueeze(0).to(device)

    answer_ems = []
    answer_ems2 = []
    with open("answers.txt") as f:
        for word in f.readlines():
            answer_ems2.append(word)
            emb = embedding_gen(word)
            pe = positional_encodings(len(word_idx), embedding_dim)
            encoded = (emb + pe).unsqueeze(0).to(device)
            _, a_vec = model(img_input, encoded)
            answer_ems.append(a_vec)

    answer_vec = torch.cat(answer_ems, dim=0)
    img_vec, ques_vec = model(img_input, question_input)
    combined_vec = img_vec + ques_vec
    logits = torch.matmul(combined_vec, answer_vec.T)
    predicted = torch.argmax(logits, dim=1)
    return answer_ems2[predicted].strip()

# ----------------- Streamlit UI -----------------
st.title("🔍 Latent-CLIP Visual Question Answering")
st.write("Upload an image and ask a question about it.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Type your question here")

if uploaded_image and question:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True, width=50)

    with st.spinner("Thinking..."):
        answer = run_inference(image, question)

    st.success(f"**Answer:** {answer}")
