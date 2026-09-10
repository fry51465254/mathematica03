# mathematica03 — S 波束缚态（课件两段代码的打靶）

按后来两段 Wolfram 片段的流水线求解图 1：先造 Gauss 动量网格，再把积分方程写成带 \(\lambda\) 的线性组，用 `LinearSolve` 抽出 \(\lambda(E_b)\)，打靶 \(\lambda(E_b)=0\)。

## 两段代码分别做什么

**未命名-2（网格）**

```wolfram
{absc, weights, errweights} = NIntegrate`GaussRuleData[nq, precision];
pt = Tan[xt * Pi/2];
jacobi = D[pt, xt];
qregion = pt /. {xt -> absc};
Δq = (jacobi /. {xt -> absc}) * weights // Abs
```

\(x\in[0,1]\) 映到 \(p=\tan(x\pi/2)\in[0,\infty)\)。现场写成 `xt -> weights` 是错的，Jacobian 必须代入节点 `absc`。

**未命名-1（打靶）**

```wolfram
eqright = λ + 4 π^2 α Sum[Δq[[j]] (p/pt) Log[...] φ[p], ...]
eq = Table[eqleft[[i]] == eqright[[i]], ...]
{b, a} = CoefficientArrays[eq, var]
λ[Eb_] := LinearSolve[a, b][[1]]
```

左端是 \((p_t^2/(2\mu)-E_b)\phi\)。右端多一个 \(\lambda\)，把齐次本征问题变成非齐次线性组。有束缚态时 \(\lambda=0\)。\(n\) 条方程、\(n+1\) 个未知数 \(\{\lambda,\phi_i\}\)，脚本里补上 \(\phi(p\sim\mu\alpha)=1\)。

对数核用图 1 的 \(\log[(p_t+p)^2+m_x^2]/[(p_t-p)^2+m_x^2]\)，前因子带 \(1/(2\pi)^3\)。现场片段把对数写反、又丢掉 \(1/(2\pi)^3\)，那样没有束缚态（\(\lambda\) 不改号）。对角奇点用课件的 \(m_x=10^{-3}\) 调节，不再做减法积分。

`p=\tan(x\pi/2)` 让动量从 \(10^{-4}\) 跨到 \(10^{4}\)，直接 `LinearSolve` 会报 `LinearSolve::luc` 病态矩阵。脚本里先做行列均衡再求解，条件数大约从 \(10^{13}\) 降到 \(10^{2}\)。

## 怎么运行

```text
Get["s_wave_coulomb.wl"]
```

默认 `mex = 10^-3`、`α = 1/137`、`μ = 1`、`nq = 160`（课件是 500；160 点已经和径向 ODE 符合到约 1%）。本环境没有 Wolfram 内核，同一套算法：

```bash
pip install -r requirements.txt
python s_wave_coulomb.py --n 160 --out results
```

## 数值结果（`nq = 160`）

Yukawa 质量 \(m_x=10^{-3}\) 会让束缚比纯库仑浅一些。

| 方法 | \(E_1\) |
|---|---|
| 库仑解析 \(-\mu\alpha^2/2\) | \(-2.664\times 10^{-5}\) |
| \(\lambda(E_b)\) LinearSolve 打靶 | \(-2.025\times 10^{-5}\) |
| Yukawa 径向 ODE 打靶 | \(-2.003\times 10^{-5}\) |

\(\lambda(E_b)\) 在基态处过零；\(\phi(p)\) 与库仑 1s 形状 \(1/(p^2+\gamma^2)^2\) 几乎重合。

## 文件

- `s_wave_coulomb.wl` — 按课件变量名写的 Wolfram 脚本
- `s_wave_coulomb.py` — 可运行的同一流水线
- `results/` — \(\lambda(E_b)\)、\(\phi(p)\)、\(u(r)\)、能级比较
