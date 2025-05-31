import json
import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def embedding_gen(sentence):
    global words
    words = sentence.lower().split()
    global word_idx
    word_idx = {word: idx for idx, word in enumerate(words)}
    global embedding_dim
    embedding_dim = 16
    global vocab_size
    global idx_only
    idx_only = [i for i in range(len(word_idx))]
    vocab_size = len(word_idx)
    embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    input_tensor = torch.LongTensor(idx_only)
    input_embeddings = embeddings(input_tensor)
    return input_embeddings

def positional_encodings(sequence_length, embedding_size):
    pe = torch.zeros(sequence_length, embedding_size)
    pos_encode = 0
    for pos in range(len(word_idx)):
        em_dim = embedding_dim
        for i in range(em_dim):
            if i%2 == 0:
                emma = torch.sin(torch.tensor(pos/(10000**((2*i)/embedding_dim))))
            else:
                emma = torch.cos(torch.tensor(pos/(10000**(((2*i)+1)/embedding_dim))))
            pe[pos][i] = emma
            #pe[pos][i+1] = emma2
    return pe


# creating the image encoder:
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.embedding_dim = 16

        # Downscaling layers for Q, K, V
        self.w_q = nn.Linear(16, 2)
        self.w_k = nn.Linear(16, 2)
        self.w_v = nn.Linear(16, 2)

        # Upscaling back to embedding dim
        self.latent_upscale = nn.Linear(2, 16)

        # Layer norm
        self.layer_norm = nn.LayerNorm(16)

        # Feedforward block
        self.feed_fwd = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16)
        )

        # Final projection
        self.output_proj = nn.Linear(16, 16)

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        context = self.latent_upscale(context)

        # Residual + Norm
        x = self.layer_norm(context + x)

        # Feedforward + Norm
        ff_out = self.feed_fwd(x)
        out = self.layer_norm(ff_out + x)

        # Final linear (optional)
        return self.output_proj(out)


# creating the text encoder:
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
        V = self.w_k(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        context = self.latent_upscale(context)

        # Residual + Norm
        x = self.layer_norm(context + x)

        # Feedforward + Norm
        ff_out = self.feed_fwd(x)
        out = self.layer_norm(ff_out + x)

        # Final linear (optional)
        return self.output_proj(out)


# defining the model:
class CLIPMini(nn.Module):
    def __init__(self):
        super(CLIPMini, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, image_patches, text_tokens):

        # Let's assume: image_patches: [B, 196, 16]
        cls_token_img = nn.Parameter(torch.randn(1, 1, 16)).to(image_patches)
        cls_token_img = cls_token_img.expand(image_patches.size(0), -1, -1)  # [B, 1, 16]

        img_input = torch.cat([cls_token_img, image_patches], dim=1)  # [B, 197, 16]
        img_embs = self.image_encoder(img_input)  # Encoder will attend to [CLS]

        cls_token_txt = nn.Parameter(torch.randn(1, 1, 16)).to(text_tokens.device)
        cls_token_txt = cls_token_txt.expand(text_tokens.size(0), -1, -1)  # [B, 1, 16]

        txt_input = torch.cat([cls_token_txt, text_tokens], dim=1)  # [B, seq_len+1, 16]
        txt_embs = self.text_encoder(txt_input)

        '''
        img_embs = self.image_encoder(image_patches)  # [B, 196, 16]
        txt_embs = self.text_encoder(text_tokens) # [B, seq_len, 16]
        '''

        # Pool
        #img_vec = torch.mean(img_embs, dim=1)      # [B, 16]
        img_vec = img_embs[:, 0, :]                # [B, 16] with CLS
        txt_vec = txt_embs[:, 0, :]                # [B, 16] with CLS

        # Normalize
        img_vec = F.normalize(img_vec, dim=-1)
        txt_vec = F.normalize(txt_vec, dim=-1)

        return img_vec, txt_vec


img_embed_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=((16, 16)), stride=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_embed_layer = img_embed_layer.to(device)

softmax = nn.Softmax(-1)

model = CLIPMini()
model.load_state_dict(torch.load("clip_mini2.pth"))
model.to(device)
model.eval()

def run_inference(image_path, question):
    # Process image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    img_patches = img_embed_layer(img).flatten(2).transpose(1, 2)  # [1, 196, 16]
    pos_mat = torch.randn(1, 196, 16).to(device)
    img_input = img_patches + pos_mat

    # Process question
    q_emb = embedding_gen(question)
    q_pe = positional_encodings(vocab_size, embedding_dim)
    question_input = (q_emb + q_pe).unsqueeze(0).to(device)

    f = open("answers.txt")
    answer_ems = []
    answer_ems2 = []
    for word in f.readlines():
        answer_ems2.append(word)
        emb = embedding_gen(word)
        pe = positional_encodings(vocab_size, embedding_dim)
        encoded = (emb + pe).to(device)
        encoded = encoded.unsqueeze(0).to(device)
        _, a_vec = model(img_input, encoded)
        answer_ems.append(a_vec)
    answer_vec = torch.cat(answer_ems, dim=0)

    img_input = img_input.to(device)
    question_input = question_input.to(device)

    img_vec, ques_vec = model(img_input, question_input)
    combined_vec = img_vec + ques_vec  # [B, dim]
    logits = torch.matmul(combined_vec, answer_vec.T)  # [B, 13]
    predicted = torch.argmax(logits, dim=1)

    #print(answer_ems2[predicted])
    return answer_ems2[predicted].strip()

run_inference("tg.jpg", "what is the color of the triangle?")