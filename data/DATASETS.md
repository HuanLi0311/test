# Natural-RNA chemical-probing datasets (out-of-distribution test pool)

Purpose: sequences that were **not** designed on Eterna, with per-nucleotide
reactivity labels, used to measure the domain shift of RibonanzaNet and its
successors. Everything here is public; no new wet-lab data.

Verified via NCBI E-utilities / GEO FTP on 2026-10-17.

---

## Tier 1 — processed reactivity profiles (directly usable)

| Accession | Organism / setting | n | Files | Size | Notes |
|---|---|---|---|---|---|
| **GSE45803** | S. cerevisiae + human, in vivo & in vitro DMS-seq | 25 | WIG tracks, RAW.tar | 384 MB | Rouskin 2014. The canonical in-vivo-vs-in-vitro pair. WIG is coverage, not reactivity — needs a ratio pipeline (vivo/vitro, or DMS/denatured). |
| **GSE153851** | SARS-CoV-2 in infected Vero cells, DMS-MaP | 6 | popavg reactivity + .ct/.dot | 0.5 MB | Lan 2022. **Already normalised reactivity**, ~30 kb single genome. Best single test case: long, natural, well-defined boundaries. |
| **GSE132099** | Human (K562/HEK293), icSHAPE in vivo + in vitro + icLASER | 14 | 4 .out.txt.gz | 82 MB | Spitale-lab style. In-cell vs in-vitro pairing at transcriptome scale. |
| **GSE146952** | Human, icSHAPE-MaP on small RNAs, in vivo + in vitro | 34 | shape.txt.gz | 2 MB | sRNA-focused; complements the mRNA sets. |
| **GSE236224** | Human neurogenesis, normalised reactivity | 9 | 1 CSV | 5 MB | Genome-wide, differentiating cells — a domain-shift stress test. |
| **GSE145805** | Mouse + human cell lines, RNA structurome | 12 | RAW.tar | 445 MB | Multi-cell-line comparison. |
| **GSE117840** | Human + mouse, subcellular compartments | 22 | RAW.tar | 108 MB | Nuclear vs cytoplasmic — folds in a localisation conditioning axis. |

## Tier 2 — single-transcript / targeted (clean but narrow)

| Accession | Target | Notes |
|---|---|---|
| GSE254361 | human primary miRNAs, SHAPE-MaP | structured, short |
| GSE245536 | telomerase RNA (TERC), in-cell DMS-MaP | lncRNA-like, 451 nt |
| GSE243328 | lncRNA SChLAP1 | long non-coding |
| GSE309440 | lncRNA DAMER + ATF4 mRNA | long non-coding |
| GSE331520 | lncRNA HOTAIR | long non-coding |
| GSE288910 | stress granules, 126 samples | largest human in-cell set found |
| GSE156671 | neuronal differentiation, 95 samples | genome-wide |
| GSE348204 | Ebola virus, icSHAPE | viral genome |

## Confirmed but not yet located

| Accession | Note |
|---|---|
| GSE164565 | Zhang 2021 in vivo SARS-CoV-2. No supplementary files at the standard suppl path — reactivity is probably in SRA only. Needs a processing pipeline; skip unless needed. |

## Known caveats to carry into the analysis

1. **Reference/annotation version drift.** Each study used its own transcriptome
   build. Reactivity must be mapped to a common coordinate system before any
   pooling, or sequence-label pairs will be misaligned.
2. **Normalisation differs.** GSE153851 ships normalised reactivity; GSE45803
   ships raw coverage. Do not mix these without re-deriving from raw.
3. **In vivo vs in vitro is confounded with study.** Different labs, different
   pipelines. Treat cross-study pooling as a batch effect, not a clean axis.
4. **Watson/Crick strand.** WIG files are split by strand; pairing plus/minus
   tracks incorrectly silently corrupts everything downstream.
