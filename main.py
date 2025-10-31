import math, random, time, torch
from torch import nn
from tqdm import tqdm

# ==============================
# 全局参数（只保留关键可调）
# ==============================
MIN_FREQ = 10
VOCAB_FILE = f"vocab_pg10_min{MIN_FREQ}.pt"
MODEL_FILE = f"pg10_transformer_v2_best_min{MIN_FREQ}.pth"

# ======================================================
# 1. 工具函数
# ======================================================
def tokenize(lines):
    return [line.strip().split() for line in lines if line.strip() != ""]

class Vocab:
    def __init__(self, tokens, min_freq=2):
        counter = {}
        for line in tokens:
            for t in line:
                counter[t] = counter.get(t, 0) + 1
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.idx_to_token += [t for t, f in self.token_freqs if f >= min_freq]
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}
    def __len__(self): return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
        return [self[token] for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[i] for i in indices]

def seq_data_iter_random(corpus, batch_size, num_steps, vocab):
    corpus = [t for line in corpus for t in line]
    corpus = [vocab[t] for t in corpus if t in vocab.token_to_idx]
    corpus = torch.tensor(corpus, dtype=torch.long)
    if len(corpus) < num_steps + 2:
        print(f"⚠️ 数据太短: tokens={len(corpus)}, num_steps={num_steps}")
        return iter(())
    num_subseqs = (len(corpus) - 1) // num_steps
    idx = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(idx)
    num_batches = num_subseqs // batch_size
    if num_batches == 0:
        print(f"⚠️ 样本过少: batch_size={batch_size}, num_steps={num_steps}")
        return iter(())
    def data(pos): return corpus[pos: pos + num_steps]
    for i in range(0, num_batches * batch_size, batch_size):
        pos = idx[i: i + batch_size]
        X = [data(p) for p in pos]
        Y = [data(p + 1) for p in pos]
        yield torch.stack(X), torch.stack(Y)

# ======================================================
# 2. 模型组件
# ======================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "P",
            torch.zeros((1, max_len, d_model), dtype=torch.float32),
            persistent=False,
        )
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        return self.dropout(X + self.P[:, :X.shape[1], :])

def subsequent_mask(size, device=None):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
    def forward(self, X): return self.net(X)

class PreNormDecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ffn, self.dropout = PositionWiseFFN(d_model, d_ff), nn.Dropout(dropout)
    def forward(self, X, mask=None):
        norm = self.ln1(X)
        out, _ = self.attn(norm, norm, norm, attn_mask=mask)
        X = X + self.dropout(out)
        X = X + self.dropout(self.ffn(self.ln2(X)))
        return X

class TransformerLanguageModelV2(nn.Module):
    def __init__(self, vocab_size, d_model=768, d_ff=3072, heads=12, layers=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList([PreNormDecoderBlock(d_model, d_ff, heads, dropout) for _ in range(layers)])
        self.ln, self.fc = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embed.weight
    def forward(self, X):
        L = X.size(1)
        mask = subsequent_mask(L, X.device)
        X = self.embed(X) * math.sqrt(self.embed.embedding_dim)
        X = self.pos(X)
        for blk in self.blocks:
            X = blk(X, mask)
        return self.fc(self.ln(X))

# ======================================================
# 3. 数据加载
# ======================================================
def load_pg10(vocab=None, batch_size=64, num_steps=128):
    with open("pg10.txt", "r", encoding="utf-8") as f:
        lines = f.read().lower().split("\n")
    tokens = tokenize(lines)
    if vocab is None:
        vocab = Vocab(tokens, min_freq=MIN_FREQ)
        torch.save(vocab, VOCAB_FILE)
    train_iter = seq_data_iter_random(tokens, batch_size, num_steps, vocab)
    return train_iter, vocab

# ======================================================
# 4. 文本生成
# ======================================================
def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / max(temperature, 1e-6)
    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(logits, top_k)
    probs = torch.softmax(values, dim=-1)
    next_token = indices[torch.multinomial(probs, 1)]
    return next_token.item()

def generate_text(model, prefix, vocab, max_len=50, device="cuda"):
    model.eval()
    tokens = [vocab[prefix]] if isinstance(prefix, str) else [vocab[p] for p in prefix]
    for _ in range(max_len):
        X = torch.tensor(tokens, device=device).unsqueeze(0)
        with torch.no_grad():
            y_hat = model(X)
        next_token = sample_next_token(y_hat[0, -1, :])
        tokens.append(next_token)
        if vocab.to_tokens(next_token) == "<eos>":
            break
    return " ".join(vocab.to_tokens(tokens))

# ======================================================
# 5. 训练主循环（去掉 warmup + 加上 PPL）
# ======================================================
def train_forever(model, vocab, device="cuda", lr=2e-4, label_smoothing=0.05):
    torch.set_float32_matmul_precision("high")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"], label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.**8)

    best, epoch = float("inf"), 0
    batch_size, num_steps = 64, 256

    print(f"⚙️ bfloat16 AMP | lr={lr} | clip=1.0 | smooth={label_smoothing}")
    while True:
        epoch += 1
        train_iter, _ = load_pg10(vocab, batch_size, num_steps)
        train_iter = list(train_iter)
        if len(train_iter) == 0:
            print("❌ 没有 batch，请增大语料或减小 batch_size/num_steps")
            break
        model.train()
        total_loss = total_tokens = 0
        start = time.time()

        for step, (X, Y) in enumerate(tqdm(train_iter, desc=f"Epoch {epoch}")):
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y_hat = model(X)
                loss = loss_fn(y_hat[:, :-1, :].reshape(-1, len(vocab)),
                               Y[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                print(f"⚠️ step {step}: loss 非法，降 lr 并跳过")
                for g in opt.param_groups: g["lr"] *= 0.9
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                print(f"⚠️ grad 爆炸: {grad_norm:.3f}")
                continue
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * Y[:, 1:].numel()
            total_tokens += Y[:, 1:].numel()
            if step % 100 == 0:
                print(f" step {step} | loss={loss.item():.4f} | grad={grad_norm:.3f}")

        avg = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg) if avg < 20 else float("inf")
        print(f"\nEpoch {epoch} | Loss={avg:.4f} | PPL={ppl:.2f} | Time={time.time()-start:.1f}s")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"💾 保存最佳模型 loss={best:.4f}, ppl={math.exp(best):.2f}")

        if epoch % 5 == 0:
            print("→ 示例:", generate_text(model, "gay", vocab, device))

# ======================================================
# 入口
# ======================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        vocab = torch.load(VOCAB_FILE, map_location="cpu")
        print(f"✅ 已加载旧 vocab（min_freq={MIN_FREQ}），共 {len(vocab)} 个词")
    except FileNotFoundError:
        print("⚠️ 未找到词表，重新生成")
        _, vocab = load_pg10(None, 64, 128)
        print(f"✅ 生成 vocab，共 {len(vocab)} 个词")

    model = TransformerLanguageModelV2(len(vocab))
    try:
        state_dict = torch.load(MODEL_FILE, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("🔁 已加载旧模型参数")
    except Exception:
        print("🚀 从零开始训练")
    train_forever(model, vocab, device)
