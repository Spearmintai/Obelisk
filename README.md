# Obelisk

**Your code runs on the new accelerator. That is not the same as running correctly.**

Moving a model from one accelerator to another is usually declared finished when
the program stops crashing. Nothing checks the part that matters: whether the
numbers coming out are still the numbers that came out before. There is no
baseline to check them against, so "it ran" becomes the whole verdict.

Obelisk supplies the missing baseline. It measures a reference device, writes the
measurements to a small file, carries that file to the target device, and reports
every place the two disagree, how far apart they are, and which hardware
assumption is responsible.

```
pip install obelisk-ai
```

## The shortest version

Run the same two lines on two machines.

```
$ obe sample          # on the NVIDIA box          $ obe sample          # on the domestic card

  device   NVIDIA GeForce RTX 4090                   device   MTT S4000
  torch    2.2.0+cu121                               torch    2.2.0

  rand (uniform)   +0.39904648 +0.51667917           rand (uniform)   +0.39904648 +0.51667917
  randn (normal)   -0.92466241 -0.42534423           randn (normal)   -0.92466241 -0.42534396
  log1p            +0.69314718 +1.09861231           log1p            NotImplementedError
  digamma          -0.57721591 +0.42278421           digamma          NotImplementedError
```

Same seed, same code, same version of PyTorch. The uniform random stream matches
bit for bit. The normal one does not. Three library functions are missing
outright. Nothing here raised a warning on the way to that difference, and
nothing in either program would have told you.

## The full version

```
# on the reference machine
obe record --out reference.obl

# copy reference.obl to the target machine — it is a small text file and it is
# everything the target needs

# on the target machine
obe verify --ref reference.obl --html report.html
```

```
  ENVIRONMENT ALIGNMENT REPORT                                     GRADE  D

  reference   NVIDIA GeForce RTX 4090   (cuda, torch 2.2.0+cu121)
  target      MTT S4000                 (musa, torch 2.2.0)

  ASSUMPTIONS
   x  lib.transcendental    Transcendental functions   ERROR      aten::log1p is missing
   !  acc.fp16_accumulate   Half-precision accumulator DIVERGENT  3.37e-04
   !  ord.atomics           Atomic accumulation order  DIVERGENT  2.40e-03
   !  lib.normalisation     Normalisation epsilon      DIVERGENT  1.32e-05
   !  rng.device_stream     Device random stream       NOT_BITWISE
   !  sel.tie_breaking      Tie breaking in sort/topk  NOT_BITWISE
   ~  acc.fp32_matmul       float32 matmul precision   WITHIN_TOLERANCE
        error_vs_exact       reference 2.93e-05   target 9.49e-05
        mantissa_bits        reference 21.7       target 20.0
```

The last two lines are a measurement rather than an accusation. Both devices are
compared against the same product computed exactly in float64 on the CPU, a
yardstick that belongs to neither of them, so each one's precision can be stated
on its own terms.

## The whole device, not just the fifteen questions

`obe record` measures the assumptions a workload depends on. `obe dump` measures
the device itself: every operator PyTorch exposes, every data type, and the
probes. The method rests on one decision. The CPU is the oracle for whether a
call is *valid*, and the device is the subject. Every case runs on the CPU first;
if it fails there, the harness called it wrong and the case is discarded rather
than counted against the device. Only a call the CPU accepts and the device
refuses is a finding.

```
obe dump --out card.obl        # on each card, once
obe diff a.obl b.obl           # anywhere, with no hardware present
```

Measured on an RTX 4090 against a Moore Threads MTT S4000, both on torch 2.2.0.
Seven hundred callables discovered, 399 valid on the CPU and run on both:

```
  OPERATOR COVERAGE  (399 operators run on both)
   behave the same                    272
   numerically different              20
   unavailable on the target          105  (73 distinct operations)

  WORKS ON THE REFERENCE, NOT ON THE TARGET
   x  std      median    svd       qr        cholesky  det       logdet
   x  asin     asinh     acosh     atanh     sinh      cosh      log1p
   x  lgamma   digamma   erfc      exp2      hypot     nansum    kthvalue
      and 52 more

  RUNS ON BOTH, RETURNS A DIFFERENT ANSWER
   !  torch.selu                         MISMATCH   4.83e-02
   !  torch.vander                       MISMATCH   1.00e+00
   !  torch.max_pool1d_with_indices      SHAPE
```

SELU multiplies positive inputs by 1.0507. On that card it returns them
unchanged, so every positive activation is five percent low, the negative branch
is correct, and nothing raises. A model built on it trains, converges, and is
wrong. That is the case no amount of running the program would have found.

Each profile carries a digest over everything it asserts, and a comparison cites
the digests it was built from, so a report can be checked against the
measurements it came from and a withdrawn measurement can be named exactly.

Profiles accumulate. Ten cards is ten hardware sessions and forty-five pairings,
and every pairing after the first is a file read.

## Making it run, and saying what that cost

A missing operator is rarely a missing capability. log1p is log(1+x), asinh is a
logarithm and a square root, a standard deviation is the square root of a
variance. The hardware can nearly always do the work; what it lacks is someone to
write the arrangement down.

```
obe run train.py
```

patches what the device cannot do and runs the program. Three kinds of
substitution, and every one of them recorded:

| | |
|---|---|
| rewrite | expressed in operators the device does have, still on the device |
| fallback | computed on the CPU, which is correct and slow, and the ledger says how slow |
| correct | the device has the operator, and the operator is wrong |

On the MTT S4000 an ordinary piece of scientific PyTorch dies at line 11 with
`aten::std.correction`. Under `obe run` it completes, having stood in for 66
operators, and finishes with this:

```
  LEDGER: what did not happen the way you wrote it

   operator                 how            calls    seconds
   selu                     correct            2      0.000
   slogdet                  fallback           1      0.001
   std                      rewrite            1      0.000
   median                   rewrite            1      0.054
   ...
   10 call(s) through 9 substituted operator(s), 0.056 s total
   0.001 s of that was spent moving data to the CPU and back
   selu was called 2 time(s); the device kernel is wrong, so the result you
   got is not the result the device would have given
```

The point is not that the program stopped crashing. Anything can stop a program
crashing. The point is the last line, which a silent compatibility layer would
never tell you.

## When the card and the code are in different places

They almost always are. The code is on a laptop or a build server; the card is in
a machine room with no route out, and nobody has put the code on it yet. That is
exactly why nobody checks.

```
obe dump --out s4000.obl              # on the card, with none of your code present
scp s4000.obl laptop:                 # one file, under a megabyte
obe check ./repo --card s4000.obl     # on a box with no accelerator and no torch
```

```
  CODE AGAINST CARD                                               GRADE  C

  code   ./normalizing-flows  (71 python files)
  card   MTT S4000  (musa)

  7 operations this code calls would fail here, and all of them can be substituted.

  WILL NOT RUN
   x  torch.slogdet         MISSING   obe can run it on the CPU   4 call sites
        normflows/flows/mixing.py:117
   x  torch.std             MISSING   obe can rewrite it          3 call sites
        normflows/flows/normalization.py:23
   x  F.logsigmoid          MISSING   obe can rewrite it          2 call sites
   x  torch.qr              MISSING   obe can run it on the CPU   2 call sites
```

Nobody bought anything, nobody shipped code into a machine room, and nobody
waited for a slot on a card.

## What it checks

Fifteen probes, each isolating one assumption that programs make about their
accelerator and never write down.

| group | assumption |
|---|---|
| accumulation | summation order, half-precision accumulator width |
| precision | float32 matmul precision, bfloat16 support |
| math library | transcendental functions, softmax numerics, normalisation epsilon |
| parallel ordering | atomic accumulation order, run-to-run stability, reduction over awkward shapes |
| randomness | the device random stream |
| selection | tie breaking in sort, topk and argmax |
| fused kernels | scaled dot-product attention, convolution |
| memory | layout and striding |

Then the same questions are asked of a real transformer: every layer of a forward
pass, and the loss and parameters across fifty training steps.

## How a difference becomes an explanation

A layer that disagrees is a symptom. The probe that disagrees in the same way is
the cause. When a layer diverges, Obelisk names the probes that could explain it
and that are themselves failing, so the report ends with sentences like *this
normalisation layer diverges, and the normalisation epsilon probe is also
diverging* rather than a wall of numbers.

## Grades

| | |
|---|---|
| **A** | bit-identical throughout |
| **B** | differs, but every difference is inside the tolerance for its dtype |
| **C** | at least one assumption is broken; usable with care |
| **D** | a critical assumption is broken, or an operation is missing |
| **E** | the device does not agree with itself |

## Commands

| | |
|---|---|
| `obe sample` | print a few numbers; run it on two machines and read them side by side |
| `obe run` | run a program with the gaps filled in, and report every substitution |
| `obe check` | judge a repository against a card profiled somewhere else |
| `obe dump` | profile this device completely: operators, dtypes, probes |
| `obe diff a.obl b.obl` | compare two profiles with no hardware present |
| `obe record` | measure the reference for one workload |
| `obe verify --ref` | compare this device against a recorded reference |
| `obe inspect` | read a repository and predict its risk, before any card exists |
| `obe migrate` | rewrite for another backend, and report what it could not change |
| `obe probes` | list the assumptions Obelisk knows how to test |
| `obe devices` | list the accelerators visible here |

A grade is never the whole answer, and the report says what drove it.

## Before you have a card

```
obe inspect ./your-repo
```

reads the source and predicts the risk from the assumptions written into it:
device names, pinned backends, hand-written kernels, precision flags,
order-dependent accumulations. On nanoGPT it finds 81 of them and returns a
predicted grade of C. The prediction is written down before the run, so it can
be held against the result afterwards.

```
obe migrate ./your-repo --target musa --patch port.diff
```

rewrites the mechanical part and, more usefully, reports what it could not safely
change, including every place the old backend is still named after the rewrite.

## Why the blueprint is a text file

Reference and target are usually not the same machine, and the interesting one
usually has no network. A blueprint is about a megabyte of JSON. It travels on a
USB stick, it can be opened and read by a person, and it carries enough of each
tensor — a digest, summary statistics, and a deterministic sample of the values —
that the two sides can be compared elementwise without either tensor ever being
in the same room as the other.

Nothing of your code or your data goes into it. Obelisk measures the device, not
the workload you are protecting.

## Tolerance

The headline number for every comparison is the largest absolute disagreement
divided by the size of the largest value in the reference. Relative error on its
own is unusable here: two values of 1e-41 that differ by 1e-41 are a hundred
percent apart and both of them are zero.

Quantities that have no excuse to differ at all — a random stream, the order of a
sort — are held to a stricter standard and must match exactly.

## Supported backends

CUDA, MUSA (Moore Threads), and CPU. A backend is three small methods; adding one
does not touch anything else.

## License

MIT.
