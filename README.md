# mathematica03 — S 波库仑束缚态（打靶 + 数值积分）

用 Mathematica 求解图 1 角积分后的 S 波动量空间方程。做法按课件后面的程序片段还原：Gauss–Legendre 数值积分、\(p=\tan(\mathrm{arc}\, x)\) 映射，再对能量打靶。坐标空间的等价径向微分方程用 `NDSolve` 再打一次，用来核对。

## 图 1 的方程

S 波、角积分之后（库仑 / Yukawa）：

\[
\left(\frac{|p_t|^2}{2\mu}-E\right)\phi(|p_t|)
=\int_0^\infty\frac{d|p|}{(2\pi)^3}(4\pi^2\alpha)\,
\frac{|p|}{|p_t|}
\log\frac{(|p_t|+|p|)^2+m_x^2}{(|p_t|-|p|)^2+m_x^2}\,
\phi(|p|)
\]

前因子 \(4\pi^2\alpha/(2\pi)^3=\alpha/(2\pi)\)。库仑极限取 \(m_x\to 0\)。原子单位 \(\mu=1,\alpha=1\) 时，解析能级是 \(E_n=-\dfrac{1}{2n^2}\)。

## 课件程序在做什么

幻灯片里的 Wolfram 片段可以还原成下面几步。

1. **数值积分网格**  
   `[0,1]` 上 5 点 Gauss–Legendre，权就是 `Out[26]`：
   `{0.118463, 0.239314, 0.284444, 0.239314, 0.118463}`。

2. **动量映射**（`In[29]`–`In[32]`）
   ```wolfram
   pt = Tan[xt * arc];
   jacobi = D[pt, xt];
   qregion = pt /. {xt -> xregion};
   Δq = (jacobi /. {xt -> xregion}) * Δx // Abs
   ```
   取 `arc = 2 ArcTan[0.904988]`，会精确对上 `Out[31]` 的
   `{0.06912, 0.35316, 0.90499, 2.1288, 5.8721}` 和 `Out[32]` 的 `Δq`。

3. **左端**  
   `(pt^2 / (2 μ) - Eb) φ[pt]`，在每个 Gauss 动量点上一份。

4. **右端**  
   把图 1 的积分换成 `Sum[Δq[[j]] * (p/pt) * Log[...] * φ[p], {j, ...}]`，再乘 `4 π^2 α`。

5. **打靶**  
   离散后是齐次问题 \((T-E)\phi=K\phi\)。调 \(E\)（或课件最后那样固定 \(E\) 调 \(\mu\)），直到 \(M(E)\phi=0\) 有非零解。归一化即可，不必引入额外的 \(\lambda\)。

## 现场代码里需要改的几处

| 问题 | 说明 |
|---|---|
| `Δp[[j]]` 深度报错 | 权重数组叫 `Δq`，不是 `Δp`。 |
| `Log` 分子分母相同 | 应使用图 1 的 \(\log[(p_t+p)^2/(p_t-p)^2]\)。后来写成 `(pt-p)/(pt+p)` 等于把核取反。 |
| 漏了 `1/(2π)^3` | 图 1 带这个因子；只写 `4 π^2 α` 会把核放大 \(8\pi^3\approx 248\) 倍。 |
| \(p=p_t\) 的 log 奇点 | 可积，但不能直接采样。\(m_x\) 过小会把对角元人为放大。本仓库用减法公式：奇点贡献改成 `NIntegrate` 拆开算的 \(J(p_t)\)。 |
| 方程里的 \(\lambda\) | 齐次本征问题不需要拉格朗日乘子；令 \(\phi_1=1\) 或看 \(\sigma_{\min}(M(E))=0\)。 |
| 5 个点太少 | 课件网格只能演示离散步骤。定量计算用 `nQuad >= 32`。 |

## 两种打靶

**动量空间（图 1）**  
Gauss 映射离散成矩阵 \(M(E)=\mathrm{diag}(p^2/2\mu-E)-K\)。打靶函数取最小奇异值 \(\sigma_{\min}(M(E))\)，在束缚能处触零。

**坐标空间（等价微分方程）**  
S 波 \(u=rR\)：

\[
u''(r)=-2\mu\bigl(E+\alpha/r\bigr)u(r),\qquad u(0)=0,\; u(\infty)=0.
\]

从 \(r=\varepsilon\) 以正则解 \(u\sim r\) 向外做数值积分（Mathematica 用 `NDSolve`，Python 用 `solve_ivp`），调 \(E\) 使 \(u(r_{\max})=0\)。

## 怎么运行

Mathematica（有许可证时）从前往后求值，或：

```text
Get["s_wave_coulomb.wl"]
```

本环境没有 Wolfram 内核，同一套算法用 Python 跑过，结果在 `results/`。

```bash
pip install -r requirements.txt
python s_wave_coulomb.py --n 32 --out results
```

默认原子单位 \(\mu=1,\alpha=1\)。课件里的 \(\alpha=1/137\) 可以 `--alpha 0.007299`，并把 `--p-scale` 设到 \(\sim\alpha\)，否则网格盖不住 \(\phi(p)\) 所在的小动量区。

## 数值结果（`n=32`，原子单位）

| n | 解析 \(E_n\) | 积分方程打靶 / Nyström | 径向 ODE 打靶 |
|---|---|---|---|
| 1 | −0.50000000 | −0.50058954 | −0.50000000 |
| 2 | −0.12500000 | −0.12555947 | −0.12500000 |
| 3 | −0.05555556 | −0.05621095 | — |
| 4 | −0.03125000 | −0.03208015 | — |

坐标空间打靶与解析值在 \(10^{-10}\) 以内。动量空间还有截断 \(p_{\max}=\tan(\mathrm{arc})\) 和 log 奇点，32 个点已经够用。加密网格会继续靠近解析能级。

## 文件

- `s_wave_coulomb.wl` — 按课件变量名写的 Wolfram 脚本（网格、`NIntegrate` 核、打靶、`NDSolve`）
- `s_wave_coulomb.py` — 可运行的同一算法，并核对课件 5 点网格
- `results/` — 打靶函数、\(\phi(p)\)、\(u(r)\)、能级比较图
