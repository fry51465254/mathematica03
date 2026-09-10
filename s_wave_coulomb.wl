(* ::Package:: *)
(*
  S-wave bound state, following the two lecture snippets:

    未命名-2  GaussRuleData + p = Tan[xt Pi/2]  ->  {qregion, Δq}
    未命名-1  eqleft == λ + 4 Pi^2 α Sum[...]   ->  λ[Eb] = LinearSolve[...][[1]]

  Then shoot FindRoot[λ[Eb] == 0].

  Get["s_wave_coulomb.wl"]
*)

ClearAll["Global`*"];


(* ========== 物理参数：与课件片段一致 ========== *)
mex = 10.^-3;
alpha = 1/137;
mu = 1;                 (* eqleft 里的约化质量；课件前面已设定 *)
nq = 160;               (* 课件 nq = 500；160 点已与 ODE 符合到 ~1% *)
precision = MachinePrecision;

(* 图 1 的前因子带 1/(2 Pi)^3。现场片段写成 4 Pi^2 α，会大 8 Pi^3 倍。 *)
includeTwoPi3 = True;
(* 图 1 的对数是 ((pt+p)^2+m^2)/((pt-p)^2+m^2)。现场片段写反了。 *)
invertLog = False;


(* ========== 未命名-2：Gauss 网格 + p = Tan[xt Pi/2] ========== *)
GaussLegendre01[n_] := Module[{nodes, weights, dP},
  nodes = Sort[x /. NSolve[LegendreP[n, x] == 0, x, Reals]];
  dP = (D[LegendreP[n, x], x] /. x -> nodes)^2;
  weights = 2/((1 - nodes^2) dP);
  {N[(nodes + 1)/2], N[weights/2]}
];

{absc, weights} = Module[{a, w, err, data},
  (* 优先用课件的 NIntegrate`GaussRuleData；失败再试 Gauss-Legendre 库。 *)
  Check[
    {a, w, err} = NIntegrate`GaussRuleData[nq, precision];
    {a, w},
    Quiet@Check[
      Needs["NumericalDifferentialEquationAnalysis`"];
      data = NumericalDifferentialEquationAnalysis`GaussianQuadratureWeights[nq, 0, 1];
      {data[[All, 1]], data[[All, 2]]},
      GaussLegendre01[nq]
    ]
  ]
];

(* x=1 时 Tan[Pi/2] 发散，把节点稍微夹住 *)
absc = Clip[N[absc], {0., 1. - 10.^-12}];

pt = Tan[xt * Pi/2];
jacobi = D[pt, xt];                    (* 先对符号 xt 求导 *)
qregion = pt /. xt -> absc;
(* 课件写成 xt -> weights，应代入节点 absc *)
Deltaq = Abs[(jacobi /. xt -> absc) * weights];

Print["nq = ", Length[qregion],
  "   pmin = ", First[qregion],
  "   pmax = ", Last[qregion]];


(* ========== 核：图 1，mex 调节对角奇点 ========== *)
logKernel[pti_, pj_] := Module[{num, den},
  num = (pti + pj)^2 + mex^2;
  den = (pti - pj)^2 + mex^2;
  If[invertLog, Log[den/num], Log[num/den]]
];

prefactor = 4 Pi^2 alpha * If[includeTwoPi3, 1/(2 Pi)^3, 1];

K = Table[
   prefactor * Deltaq[[j]] * (qregion[[j]]/qregion[[i]]) *
     logKernel[qregion[[i]], qregion[[j]]],
   {i, Length[qregion]}, {j, Length[qregion]}];


(* ========== 未命名-1：eqleft == λ + Sum，LinearSolve 得 λ[Eb] ========== *)
(*
  现场片段：
    eqright = λ + 4 π^2 α Sum[Δq[[j]] (p/pt) Log[...] φ[p]]
    eq = eqleft == eqright
    {b, a} = CoefficientArrays[eq, var]
    λ[Eb_] := LinearSolve[a, b][[1]]

  n 条物理方程、未知数 {λ, φ1..φn} 共 n+1 个，必须加一条归一化。
  下面用 φ 在 p ~ μ α 处等于 1，与 LinearSolve 抽取 λ 完全同构。
*)
i0 = First@Ordering[Abs[qregion - mu alpha], 1];

(* p 从 ~10^-4 跨到 ~10^4，直接 LinearSolve 会报 luc 病态矩阵。
   先按行、列无穷范数把 A 均衡到 O(1)，条件数可从 ~10^13 降到 ~10^2。 *)
scaledLinearSolve[A_?MatrixQ, rhs_?VectorQ] := Module[
  {rowS, colS, A1, A2, rhs1, y},
  rowS = Max[#, 10.^-30] & /@ (Max /@ Abs[A]);
  A1 = MapThread[#1/#2 &, {A, rowS}];
  rhs1 = rhs/rowS;
  colS = Max[#, 10.^-30] & /@ (Max /@ Abs[Transpose[A1]]);
  A2 = A1 . DiagonalMatrix[1/colS];
  y = Quiet[LinearSolve[A2, rhs1], {LinearSolve::luc}];
  y/colS
];

assembleSystem[Eb_?NumericQ] := Module[{n, A, rhs, T},
  n = Length[qregion];
  A = ConstantArray[0., {n + 1, n + 1}];
  rhs = ConstantArray[0., n + 1];
  T = qregion^2/(2 mu) - Eb;
  (* (T - K) φ - λ == 0 *)
  A[[1 ;; n, 1]] = -1.;
  A[[1 ;; n, 2 ;; n + 1]] = DiagonalMatrix[T] - K;
  A[[n + 1, i0 + 1]] = 1.;
  rhs[[n + 1]] = 1.;
  {A, rhs}
];

lambdaOfEb[Eb_?NumericQ] := Module[{A, rhs},
  {A, rhs} = assembleSystem[Eb];
  scaledLinearSolve[A, rhs][[1]]
];

phiOfEb[Eb_?NumericQ] := Module[{A, rhs, sol},
  {A, rhs} = assembleSystem[Eb];
  sol = scaledLinearSolve[A, rhs];
  {sol[[1]], sol[[2 ;;]]}
];

(* 小 n 时按课件用 CoefficientArrays，核对与矩阵 LinearSolve 一致 *)
Module[{nDemo = 8, abscD, wD, qD, dD, KD, i0D, eqleft, eqright, polys, var, b, a, EbN, lamCA, lamM},
  {abscD, wD} = GaussLegendre01[nDemo];
  qD = Tan[abscD Pi/2];
  dD = Abs[(Pi/2) (1 + qD^2) wD];
  KD = Table[prefactor dD[[j]] (qD[[j]]/qD[[i]]) logKernel[qD[[i]], qD[[j]]],
    {i, nDemo}, {j, nDemo}];
  i0D = First@Ordering[Abs[qD - mu alpha], 1];
  EbN = -mu alpha^2/4;
  eqleft = Table[(qD[[i]]^2/(2 mu) - EbN) φ[i], {i, nDemo}];
  eqright = Table[
    λ + Sum[KD[[i, j]] φ[j], {j, nDemo}], {i, nDemo}];
  polys = Join[eqleft - eqright, {φ[i0D] - 1}];
  var = Join[{λ}, Table[φ[i], {i, nDemo}]];
  {b, a} = CoefficientArrays[polys, var];
  lamCA = LinearSolve[a, -b][[1]];
  (* 同一套方程的矩阵形式 *)
  Module[{A, rhs},
    A = ConstantArray[0., {nDemo + 1, nDemo + 1}];
    rhs = ConstantArray[0., nDemo + 1];
    A[[1 ;; nDemo, 1]] = -1.;
    A[[1 ;; nDemo, 2 ;;]] = DiagonalMatrix[qD^2/(2 mu) - EbN] - KD;
    A[[nDemo + 1, i0D + 1]] = 1.;
    rhs[[nDemo + 1]] = 1.;
    lamM = scaledLinearSolve[A, rhs][[1]];
  ];
  Print["CoefficientArrays vs matrix λ[", EbN, "] = ", {lamCA, lamM}];
];


(* ========== 打靶：FindRoot[λ[Eb] == 0] ========== *)
Ecoul = -mu alpha^2/2;
Print["Coulomb analytic E1 = ", Ecoul];

scanE = Table[e, {e, 2. Ecoul, -10.^-7, ( -10.^-7 - 2. Ecoul)/24}];
scanLam = lambdaOfEb /@ scanE;
Do[
  Print["  Eb = ", ScientificForm[scanE[[i]], 4],
    "   λ = ", ScientificForm[scanLam[[i]], 4]],
  {i, 1, Length[scanE], 4}];

eShot = Eb /. FindRoot[lambdaOfEb[Eb] == 0, {Eb, 1.2 Ecoul, 0.4 Ecoul},
    AccuracyGoal -> 8, PrecisionGoal -> 8];
{lamShot, phiNum} = phiOfEb[eShot];
Print["λ-shot E1 = ", eShot, "   λ = ", lamShot];


(* ========== 坐标空间 Yukawa ODE 打靶（核对） ========== *)
rStart = 10.^-8;
rMax = 25./(mu alpha); (* ~ 25 a0 *)

uEnd[E0_?NumericQ] := Module[{u, sol, rstop},
  sol = Quiet@NDSolve[
     {u''[r] == -2 mu (E0 + alpha Exp[-mex r]/r) u[r],
      u[rStart] == rStart, u'[rStart] == 1,
      WhenEvent[Abs[u[r]] > 10^8, "StopIntegration"]},
     u, {r, rStart, rMax}, MaxSteps -> 10^6];
  If[sol === {}, 10.^8,
    rstop = (u /. sol[[1]])["Domain"][[1, 2]];
    (u /. sol[[1]])[rstop]]
];

eODE = e /. FindRoot[uEnd[e] == 0, {e, 1.2 Ecoul, 0.4 Ecoul}];
Print["ODE-shot E1  = ", eODE];

uSol = NDSolveValue[
   {u''[r] == -2 mu (eODE + alpha Exp[-mex r]/r) u[r],
    u[rStart] == rStart, u'[rStart] == 1},
   u, {r, rStart, Min[rMax, 8./(mu alpha)]}];


(* ========== 图 ========== *)
gamma = mu alpha;
plotLam = ListLinePlot[Transpose[{scanE, scanLam}],
   AxesLabel -> {"Eb", "λ"}, PlotLabel -> "λ(Eb) shooting function",
   Mesh -> All, Epilog -> {Dashed, InfiniteLine[{eShot, 0}, {0, 1}]}];

plotPhi = ListLinePlot[
   {Transpose[{qregion, phiNum/Max[Abs[phiNum]]}],
    Transpose[{qregion, (1/(qregion^2 + gamma^2)^2)/Max[1/(qregion^2 + gamma^2)^2]}]},
   PlotRange -> {{0, 8 gamma}, All},
   PlotLegends -> {"λ-shot φ(p)", "Coulomb 1s shape"},
   AxesLabel -> {"p", "φ"}, PlotLabel -> "Momentum-space ground state"];

a0 = 1/(mu alpha);
plotU = Plot[uSol[r]/uSol[a0], {r, rStart, 6 a0},
   AxesLabel -> {"r", "u"}, PlotLabel -> "Yukawa radial u(r) from ODE shooting"];

Print["Done. Evaluate plotLam, plotPhi, plotU."];
plotLam
plotPhi
plotU
