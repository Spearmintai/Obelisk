# Obelisk

**Your code runs on the new accelerator. That is not the same as running correctly.**

[![PyPI](https://img.shields.io/pypi/v/obelisk-env)](https://pypi.org/project/obelisk-env/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/project/obelisk-env/)

[中文](README.zh-CN.md)

```
pip install obelisk-env
```

---

## 1. The problem

Run the same two lines on two machines. Same seed, same code, same version of PyTorch.

| | NVIDIA RTX 4090 | Moore Threads MTT S4000 |
|---|---|---|
| `rand` (uniform) | +0.51667917 | +0.51667917 |
| `randn` (normal) | −0.42534423 | −0.4253**4396** |
| `log1p` | +0.69314718 | `NotImplementedError` |

The uniform random stream matches bit for bit, so the generator was ported correctly. Everything downstream of it is another matter.

Then there is this, which is worse because nothing breaks. `SELU` is a standard activation function that multiplies positive inputs by 1.0507.

| computed by | result | error |
|---|---|---|
| the closed-form definition, in float64 | 3.216856 | — |
| CPU | 3.216856 | 1.2e−07 |
| RTX 4090 | 3.216856 | 1.2e−07 |
| **MTT S4000** | **3.061628** | **4.8e−02** |

No exception, no warning, no degraded mode. A model built on SELU trains on this card, produces a loss curve, converges, and is wrong by five percent in every positive activation.

This is the gap the industry does not measure. There are many tools for *will it run*. There is nothing for *is it right*. And a buyer signing for a hundred million yuan of compute needs the second one.

## 2. The product

Three questions, each needing less access than the last.

| question | what it needs | size | who gets it |
|---|---|---|---|
| **Will my code run?** | operator and dtype support lists | 3 KB | publishable |
| **Will it run correctly?** | numerical fingerprints | 562 KB | the buyer |
| **What did it actually do?** | the substitution ledger | at runtime | the operator |

The method rests on four decisions.

- **The CPU is the oracle for validity; the device is the subject.** Every case runs on the CPU first. If it fails there, the harness called it wrong and the case is discarded rather than counted against the device.
- **A fingerprint is a digest, float64 statistics, and 512 deterministic samples.** Small enough to travel in a text file, rich enough to compare elementwise with neither tensor present.
- **Error is measured against the scale of the tensor,** not elementwise. Two values of 1e-41 that differ by 1e-41 are a hundred percent apart and both of them are zero.
- **Quantities with no excuse to differ are held to bitwise equality:** a random stream, the order of a sort. Arithmetic only has to be close.

```
obe sample     print a few numbers; run it on two machines and read them side by side
obe dump       profile a device completely: every operator, every dtype, 15 probes
obe diff       compare two profiles, with no hardware present
obe cards      the devices this copy already knows about
obe check      judge a repository against a card profiled somewhere else
obe run        run a program with the gaps filled in, and report every substitution
obe record     measure a reference for one workload
obe verify     compare this device against a recorded reference
obe inspect    read a repository and predict its risk, before any card exists
obe migrate    rewrite for another backend, and report what it could not change
```

## 3. The demonstration

**Thirty seconds.** One command on each machine, then read the two outputs side by side.

```
obe sample
```

**The full version, where the card and the code never meet.** This is the arrangement the work actually takes: the code is on a laptop, the card is in a machine room with no route out, and nobody has put one on the other.

```
# on the card, with none of your code present
obe dump --out s4000.obl

# one file, under a megabyte, carried by whatever means the machine room allows
scp s4000.obl laptop:

# on a box with no accelerator and no PyTorch installed
obe check ./your-repo --card s4000.obl
```

```
  CODE AGAINST CARD                                              GRADE  C

  7 operations certainly and 1 probably would fail here,
  and every one can be substituted.

   x  torch.slogdet    4 call sites   normflows/flows/mixing.py:117
   x  torch.std        3 call sites   normflows/flows/normalization.py:23
   x  F.logsigmoid     2 call sites   normflows/transforms.py:29
   x  torch.qr         2 call sites   normflows/flows/mixing.py:73
```

Nobody bought a card, nobody shipped code into a machine room, nobody waited for a slot.

**Then make it run anyway, and say what that cost.**

```
obe run train.py
```

A missing operator is rarely a missing capability. `log1p` is `log(1+x)`; a standard deviation is the square root of a variance. Obelisk fills them in three ways and records every one: **rewritten** in operators the device does have, sent back to the **CPU**, or **corrected** because the device's own kernel is wrong.

```
  LEDGER: what did not happen the way you wrote it

   operator      how          calls   seconds    accuracy
   selu          correct          2     0.000       exact
   slogdet       fallback         1     0.001       exact
   std           rewrite          1     0.000       exact
   log1p         fallback         1     0.008     3.1e-01

   selu was called 2 times; the device kernel is wrong, so the result you
   got is not the result the device would have given
```

The point is not that the program stopped crashing. Anything can stop a program crashing. The point is the last line, which a silent compatibility layer would never tell you.

## 4. The validation

Measured on 12 October 2026, on two rented machines, with PyTorch 2.2.0 on both sides so the framework version is not part of the difference.

**Operator coverage.** Seven hundred callables discovered by introspection; 399 accepted by the CPU and put to both cards.

| | RTX 4090 | MTT S4000 |
|---|---|---|
| behave the same | 386 | 272 |
| return a different answer | 10 | 20 |
| **do not exist** | **1** | **73 distinct operations** |

Absent on the domestic card, among others: `std`, `median`, `svd`, `qr`, `cholesky`, `det`, `logdet`, `slogdet`, `lu`, `triangular_solve`, `cdist`, `pdist`, `asin`, `asinh`, `acosh`, `atanh`, `sinh`, `cosh`, `log1p`, `exp2`, `erfc`, `lgamma`, `digamma`, `i0`, `nansum`, `nanmean`, `kthvalue`, `logaddexp`, `logsigmoid`, `mish`.

**Wrong answers, raised silently.** `SELU` low by 5%. `torch.log` wrong by 30.7% at 1+2⁻²³, 4.0% at 1+2⁻²⁰, and accurate to 2.9e−08 at 1.5, which poisons any log-likelihood landing near one. `vander` wrong by 100%. `max_pool1d_with_indices` returns a different shape.

**Data types.** float64 allocates and sums but cannot add or multiply. complex64 cannot allocate at all.

**Assumptions.** Of fifteen probes, three diverge beyond tolerance: half-precision accumulator width at 3.37e−04, layer_norm epsilon at 1.32e−05, scatter-add ordering at 4.33e−03. Two that should match exactly do not: the random stream, where the uniform draw is bit-identical and the normal transform is not, and tie-breaking in sort, topk and argmax.

**One measurement in the card's favour.** float32 matmul carries 20.0 effective mantissa bits against 21.7 on the 4090, measured against the same product computed exactly in float64 on a CPU. A shortened float32 would show 11. Neither card is cheating, and the measurement says so in a form anyone can repeat.

**And the honest negative, which is what makes the rest believable.** A four-layer transformer ran on both cards with identical weights and batches: 28 of 28 layers inside tolerance, final loss after fifty steps 3.73e−09 apart, parameters 5.36e−09 apart. This workload touches none of the broken parts. It is fine on this card. Naming which workloads are affected and which are not is the whole job; a report that only ever says no is not worth reading.

**We ran the same scrutiny over our own instrument.** The first set of substitutes committed the exact fault this tool exists to catch: `log(1+x)` returns zero for small x, so our `log1p` was wrong by 100% and silent. They are now written the way a library writes them, and every substitute measures itself at startup against the operator in float64, refusing to install if it is worse than 1e−4. That rule immediately earned its place: the corrected `log1p` still measured 3.1e−01 on the S4000, while the identical code on a CPU measures 1.2e−07 and on a 4090 matches the CPU bit for bit. The formula was right. The card's logarithm was not. That is how the `log` defect above was found.

## 5. Where it stands

- **On PyPI.** `pip install obelisk-env`, installs into an empty environment, no dependencies.
- **Tested against seven real repositories**, including diffusers at 2,125 Python files, read in about a second with no accelerator and no PyTorch.
- **Two devices in the card library**, shipped inside the package, so `obe check ./repo --card mtt-s4000-musa` works with no file at all.
- **The three-machine workflow is proven end to end**: card profiled alone, file carried, verdict produced on a box with no GPU.

### Boundaries

Everything measured here is specific to one Moore Threads MTT S4000 under driver 2.7.0 and torch 2.2.0. No Ascend, Cambricon or Iluvatar card has been tested. A card is only true of the configuration it was measured on, which is why every card records its driver and framework versions and `obe check` prints them. A stale card is worse than no card, because it is believed.

## License

MIT.
