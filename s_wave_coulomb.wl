(* ::Package:: *)
(*
  S-wave Coulomb bound state
  --------------------------
  图 1（角积分后的 S 波动量空间方程）:

    ( |pt|^2 / (2 mu) - E ) phi(|pt|)
      = Integral dp/(2 Pi)^3 * (4 Pi^2 alpha) * (|p|/|pt|)
        * Log[ ((|pt|+|p|)^2 + mex^2) / ((|pt|-|p|)^2 + mex^2) ] phi(|p|)

  课件做法（由后续程序片段还原）:
    1. 在 x in [0,1] 上做 Gauss-Legendre 数值积分;
    2. 动量映射 p = Tan[arc * x]，权重乘上 Jacobian;
    3. 把积分方程离散成矩阵，打靶（调 E 使齐次方程有非平凡解）.

  用法（Mathematica / wolframscript）:
      Get["s_wave_coulomb.wl"]
    或在 Notebook 中从前往后逐段求值.
*)

ClearAll["Global`*"];


(* ========== 1. 物理参数（原子单位：氢原子 E_n = -1/(2 n^2)） ========== *)
(* 课件里 alpha = 1/137。要复现氢原子标准能级，先用原子单位 alpha = 1, mu = 1。
   换成物理单位时设 alpha = 1/137，并把动量网格缩到 p ~ mu*alpha。 *)
mu = 1;
alpha = 1;
mex = 0;                 (* 库仑：mex -> 0；有限 mex 对应 Yukawa *)
nQuad = 32;              (* Gauss 点数；课件演示用 5，定量计算建议 >= 32 *)
arc = 1.55;              (* 映射 p = Tan[arc x]，arc < Pi/2，使 p 有限 *)
pScale = 1;              (* p = pScale * Tan[arc x]；alpha = 1/137 时可取 pScale = alpha *)


(* ========== 2. [0,1] 上的 Gauss-Legendre（对应课件 Out[26] 的 Delta x） ========== *)
GaussLegendre01[n_] := Module[{nodes, weights, deriv2},
  nodes = Sort[x /. NSolve[LegendreP[n, x] == 0, x, Reals]];
  deriv2 = (D[LegendreP[n, x], x] /. x -> nodes)^2;
  weights = 2/((1 - nodes^2) deriv2);
  {N[(nodes + 1)/2], N[weights/2]}
];

{xregion, dx} = GaussLegendre01[nQuad];

(* 课件 5 点权重复现：{0.118463, 0.239314, 0.284444, 0.239314, 0.118463} *)
{x5, dx5} = GaussLegendre01[5];
Print["5-point Gauss weights (lecture Out[26]) = ", dx5];


(* ========== 3. 动量映射 p = pScale * Tan[arc x]（课件 In[29]--In[32]） ========== *)
(* 必须先对符号 xt 求导，再代入数值；不要写成 D[pOfx[0.5], 0.5]。 *)
pOfx[xt_] := pScale * Tan[xt * arc];
jacobiExpr = D[pScale * Tan[xt * arc], xt]; (* pScale * arc * Sec[arc xt]^2 *)

qregion = pOfx /@ xregion;
Deltaq = Abs[jacobiExpr /. xt -> xregion] * dx;  (* 课件里的 Delta q；注意不是 Delta p *)
pmax = pOfx[1.];                                (* x = 1 对应的动量截断 *)

(* 课件 5 点网格：arc 取 2 ArcTan[0.904988] 即可对上 Out[31], Out[32] *)
Module[{arcL, p5, jac5, dq5},
  arcL = 2 ArcTan[0.904988];
  p5 = Tan[arcL * x5];
  jac5 = arcL (1 + p5^2);
  dq5 = Abs[jac5] * dx5;
  Print["lecture qregion ~ ", p5];
  Print["lecture Deltaq  ~ ", dq5];
];

Print["working grid: n = ", nQuad, ", pmax = ", pmax];
Print["qregion (first/last) = ", {First[qregion], Last[qregion]}];


(* ========== 4. 角积分核（图 1 / 有质量推广） ========== *)
(* 正确的对数与课件图 1 一致，不要写成分子分母相同，也不要写反。 *)
kappa[pt_?NumericQ, p_?NumericQ] := Module[{num, den},
  num = (pt + p)^2 + mex^2;
  den = (pt - p)^2 + mex^2;
  (p/pt) Log[num/den]
];

(* 图 1 的前因子：4 Pi^2 alpha / (2 Pi)^3 = alpha / (2 Pi)
   课件现场代码漏掉了 1/(2 Pi)^3，核会大 8 Pi^3 ~ 248 倍。 *)
prefactor = 4 Pi^2 alpha / (2 Pi)^3;


(* ========== 5. 用数值积分构造矩阵（奇点用减法公式） ========== *)
(*
  被积函数在 p = pt 处有可积的 log 奇点。
  直接把 Gauss 对角元写成 Log[(2p)^2 / 0] 会炸；mex 取得太小则对角被任意放大。

  减法：
    Integral kappa(pt,p) phi(p) dp
      = Integral kappa(pt,p) (phi(p)-phi(pt)) dp + phi(pt) * J(pt)
  第一项在 p = pt 处为 0，对角可安全地丢掉；
  J(pt) = Integral_0^{pmax} kappa(pt,p) dp 用 NIntegrate 在 pt 处拆开计算。
*)
Jint[pt_?NumericQ] :=
  NIntegrate[kappa[pt, p], {p, 0, pt},
      PrecisionGoal -> 8, AccuracyGoal -> 8, MaxRecursion -> 20] +
   NIntegrate[kappa[pt, p], {p, pt, pmax},
      PrecisionGoal -> 8, AccuracyGoal -> 8, MaxRecursion -> 20];

Print["building kernel (NIntegrate for J, n = ", nQuad, ") ..."];
Jvals = Table[Jint[qregion[[i]]], {i, nQuad}];

Kreg = Table[
   If[i == j, 0.,
    prefactor * Deltaq[[j]] * kappa[qregion[[i]], qregion[[j]]]],
   {i, nQuad}, {j, nQuad}];

K = Kreg;
Do[
  K[[i, i]] = prefactor * Jvals[[i]] - Total[Kreg[[i]]],
  {i, nQuad}];

Tdiag = qregion^2 / (2 mu);


(* ========== 6. 打靶法：调 E，使齐次方程有非平凡解 ========== *)
(*
  (T - E) phi = K phi
  M(E) phi = 0,   M(E) = diag(p^2/(2 mu) - E) - K
  打靶函数取最小奇异值 sigma_min(M(E))；束缚能处它触零（不改号）。
  这就是把积分方程离散之后的打靶，对应课件最后用方程列表求根。
  不要额外引入 lambda：齐次问题用归一化（或 SVD）即可，不需要拉格朗日乘子。
*)
Mmat[E_?NumericQ] := DiagonalMatrix[Tdiag - E] - K;
shootSigma[E_?NumericQ] := First[SingularValueList[Mmat[N[E]], 1]];

Print["\n===== shooting scan of sigma_min(E) ====="];
scanE = Table[e, {e, -0.70, -0.02, 0.04}];
scanS = shootSigma /@ scanE;
Do[
  Print["  E = ", PaddedForm[scanE[[i]], {6, 3}],
    "   sigma_min = ", ScientificForm[scanS[[i]], 4]],
  {i, Length[scanE]}];

(* sigma_min >= 0，只在本征值处触零，用 FindMinimum 打靶比 FindRoot 稳。 *)
eShot = e /. Last[FindMinimum[shootSigma[e], {e, -0.50},
     AccuracyGoal -> 10, PrecisionGoal -> 10]];
Print["\nshot ground-state energy E1 = ", eShot];
Print["analytic E1 = ", -mu alpha^2 / 2];

eShot2 = e /. Last[FindMinimum[shootSigma[e], {e, -0.125},
     AccuracyGoal -> 10, PrecisionGoal -> 10]];
Print["shot E2 = ", eShot2, "   analytic = ", -mu alpha^2 / (2 2^2)];


(* ========== 7. 对照：直接把 (T - K) 当本征问题 ========== *)
{evals, evecs} = Eigensystem[N[DiagonalMatrix[Tdiag] - K]];
ord = Ordering[Re[evals]]; (* 最深束缚在前 *)
evals = Re[evals[[ord]]];
evecs = Re[evecs[[ord]]];
Print["\n===== Nyström eigenvalues (lowest 5) ====="];
Print[evals[[1 ;; 5]]];
Print["analytic {E1,E2,E3,E4} = ",
  Table[-mu alpha^2 / (2 n^2), {n, 1, 4}]];


(* ========== 8. 动量空间波函数（基态）并与 1s 解析形状比较 ========== *)
phiNum = evecs[[1]];
(* 解析 1s：phi(p) ~ 1/(p^2 + (mu alpha)^2)^2 ，只比形状 *)
gamma = mu alpha;
phiAn = 1/(qregion^2 + gamma^2)^2;
scalePhi = Max[Abs[phiNum]]/Max[phiAn];
phiAn *= scalePhi * Sign[phiNum[[Ordering[Abs[phiNum], -1][[1]]]]];


(* ========== 9. 等价径向微分方程的打靶（NDSolve 数值积分） ========== *)
(*
  S 波 u(r) = r R(r):
      u''(r) = -2 mu (E + alpha/r) u(r)
      u(0) = 0,  u(Infinity) = 0
  从 r = eps 以正则解 u ~ r 向外积分，调 E 使 u(rmax) = 0。
  这是同一物理问题在坐标空间的打靶，用来核对接动量空间结果。
*)
rStart = 10.^-6;
rMax = 40.;

uEnd[E0_?NumericQ] := Module[{u, sol, rstop},
  sol = Quiet@NDSolve[
     {u''[r] == -2 mu (E0 + alpha/r) u[r],
      u[rStart] == rStart, u'[rStart] == 1,
      WhenEvent[Abs[u[r]] > 10^8, "StopIntegration"]},
     u, {r, rStart, rMax},
     MaxSteps -> 10^6];
  If[sol === {}, 10.^8,
    rstop = (u /. sol[[1]])["Domain"][[1, 2]];
    (u /. sol[[1]])[rstop]
  ]
];

Print["\n===== coordinate-space ODE shooting ====="];
eODE = e /. FindRoot[uEnd[e] == 0, {e, -0.6, -0.3}];
Print["ODE-shot E1 = ", eODE, "   analytic = ", -mu alpha^2 / 2];

eODE2 = e /. FindRoot[uEnd[e] == 0, {e, -0.16, -0.08}];
Print["ODE-shot E2 = ", eODE2, "   analytic = ", -mu alpha^2 / (8)];

(* 画出打到的基态径向波函数 *)
uSol = NDSolveValue[
   {u''[r] == -2 mu (eODE + alpha/r) u[r],
    u[rStart] == rStart, u'[rStart] == 1},
   u, {r, rStart, rMax}];


(* ========== 10. 图 ========== *)
plotSigma = ListLinePlot[Transpose[{scanE, scanS}],
   PlotRange -> All, Mesh -> All,
   AxesLabel -> {"E", "sigma_min"},
   PlotLabel -> "Shooting function of the integral operator",
   Epilog -> {Dashed, InfiniteLine[{eShot, 0}, {0, 1}]}];

plotPhi = ListLinePlot[
   {Transpose[{qregion, phiNum/Max[Abs[phiNum]]}],
    Transpose[{qregion, phiAn/Max[Abs[phiAn]]}]},
   PlotLegends -> {"Nyström phi(p)", "analytic 1s shape"},
   AxesLabel -> {"p", "phi"}, PlotRange -> {{0, 8}, All},
   PlotLabel -> "Momentum-space S-wave (ground state)"];

plotU = Plot[{uSol[r]/uSol[1.], 2 r Exp[-r]/(2 Exp[-1.])},
   {r, rStart, 12},
   PlotLegends -> {"ODE shooting u(r)", "analytic 2 r e^{-r}"},
   AxesLabel -> {"r", "u"},
   PlotLabel -> "Coordinate-space radial wave function"];

Print["\nDone. Evaluate plotSigma, plotPhi, plotU to see the figures."];
plotSigma
plotPhi
plotU
