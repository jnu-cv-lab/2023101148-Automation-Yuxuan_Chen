import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. Sinusoidal Position Encoding ======================
def sinusoidal_position_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, None]                # (max_len, 1)
    i = np.arange(d_model)[None, :]                  # (1, d_model)
    angle_rate = 1.0 / np.power(10000, (2 * (i // 2)) / d_model)
    angle = pos * angle_rate
    pe = np.zeros_like(angle)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

# ====================== 2. 二维向量旋转 ======================
def rotate_2d(v, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([v[0] * cos_t - v[1] * sin_t,
                     v[0] * sin_t + v[1] * cos_t])

# ====================== 3. 高维 RoPE ======================
def rope(x, pos, theta_base=10000.0):
    d = len(x)
    assert d % 2 == 0, "Dimension must be even"
    i = np.arange(d // 2)
    theta = 1.0 / (theta_base ** (2 * i / d))
    angle = pos * theta
    y = x.copy()
    for i in range(d // 2):
        idx = 2 * i
        y[idx], y[idx + 1] = rotate_2d(x[idx:idx+2], angle[i])
    return y

# ====================== 4. E+pos 与 RoPE 输入对比 ======================
def e_plus_pos(word_emb, pos_enc):
    return word_emb + pos_enc

# ====================== 5. 数值实验验证 RoPE 相对位置性质 ======================
def verify_rope_relative():
    d = 64
    q = np.random.randn(d)
    k = np.random.randn(d)
    pairs = [(0, 0), (1, 2), (5, 6), (2, 1), (6, 5)]  # diffs: 0, -1, -1, 1, 1
    print("=== RoPE Relative Position Property Verification ===")
    print("m, n, diff, dot product")
    for m, n in pairs:
        q_m = rope(q, m)
        k_n = rope(k, n)
        score = np.dot(q_m, k_n)
        print(f"{m}, {n}, {m-n:2d}, {score:.6f}")
    # Plot
    diffs = np.arange(-10, 11)
    scores = []
    for delta in diffs:
        m = 0
        n = m - delta
        q_m = rope(q, m)
        k_n = rope(k, n)
        scores.append(np.dot(q_m, k_n))
    plt.figure(figsize=(8,4))
    plt.plot(diffs, scores, 'b-o')
    plt.xlabel('Relative position (m - n)')
    plt.ylabel('Dot product')
    plt.title('RoPE: Dot product depends on relative position')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rope_relative.png')
    plt.close()
    print("Plot saved as rope_relative.png")

# ====================== 主程序 ======================
if __name__ == "__main__":
    print("1. Sinusoidal position encoding (seq_len=10, d=8):")
    pe = sinusoidal_position_encoding(10, 8)
    print("Shape:", pe.shape)
    print(pe[:2])

    print("\n2. 2D rotation: vector [1,0] rotated by 45° ->", rotate_2d([1,0], np.pi/4))

    print("\n3. High-dimensional RoPE: vector [0,1,...,7] at position 2")
    x = np.arange(8, dtype=float)
    x_rope = rope(x, 2)
    print("Before:", x)
    print("After: ", x_rope)
    print("Norm preserved:", np.linalg.norm(x), np.linalg.norm(x_rope))

    print("\n4. E+pos vs RoPE input:")
    emb = np.array([0.5, -0.2, 1.3, 0.7])
    pos_enc = np.array([0.1, 0.3, 0.2, 0.0])
    print("E+pos:", e_plus_pos(emb, pos_enc))
    print("RoPE (pos=1):", rope(emb, 1))

    print("\n5. Numerical verification of RoPE relative position property:")
    verify_rope_relative()