# Thesis Final-Revision Review — SE-CDT

**Date:** 2026-07-27
**Subject:** `report/HCMUT_Master_Thesis_Template/` (the deliverable — 126 pp as reviewed, 127 pp after the P0 fixes)
**Checked against:** `BIỂU MẪU 4` (layout / front-matter order) and `BIỂU MẪU 5` (task sheet)
**Scope:** citation validity, format compliance, internal consistency of the methodology write-up

> **The methodology is frozen.** Every methodology item below is a *description or consistency* fix in the text — not a change to the method, the code, or the experiments. The one item that would require new computation is explicitly marked **optional**.

## How these findings were established

- Both official templates extracted and read: `antiword -m UTF-8` for the `.doc`, `word/document.xml` for the `.docx`.
- `report/HCMUT_Master_Thesis_Template/main.tex` rebuilt from scratch (pdflatex → bibtex → pdflatex ×2). Build artifacts and the overwritten `main.pdf` were reverted afterwards; the repo was left clean.
- Every number in the prose cross-checked against the generated `results/tables/*.tex`. **Those tables are auto-generated, so they are treated as ground truth and the prose is what needs correcting.**
- Scripted cite/label integrity check across `chapters/`, `ext_pages/`, `results/tables/`, `main.tex`.
- Pages rendered to PNG (PyMuPDF) to confirm every layout claim visually: cover p1, task sheet p6, abbreviations p21, conclusion table p111, publications p117, VITA p126.
- Risky bib entries verified against the Crossref API and JMLR.

> **Retracted false positives — do not chase these.** `pdftotext -layout` garbles nested `p{}` tabulars. It made the Vietnamese task-sheet table and Table 6.1 look badly misaligned. Rendering both pages to PNG showed they are **correct**. Always render to image before believing a table is broken.

## 1. Verified state

| Check | As reviewed | After P0 fixes |
|---|---|---|
| Build | 0 errors, 0 undefined refs, 0 undefined citations | unchanged |
| Output | 126 pages, A4 (595.276 × 841.89 pt) | **127 pages**, A4 |
| Typesetting warnings | 20 overfull hbox, 15 underfull hbox | unchanged |
| Printed references | 49 `\bibitem`s | **56** (+Demšar, +6 Kafka) |
| Bib entries defined | 99 (so 50 never printed) | 99 (43 still never print) |
| `\ref` integrity | all resolve; no duplicate labels | unchanged |

Structurally the document is sound. Everything below is content-level.

### Single source of truth (resolved 2026-07-27)

There used to be a second, diverged thesis tree in `report/latex/` (107 pp, Vietnamese headings, different bib — it had `xiang2023review` where the deliverable has `celik2020adaptation`, and it cited `wang2020kswin` for KSWIN where the deliverable cites only `raab2020reactive`). Two PDFs gave two different answers for the same citation.

**Resolved:** the duplicate thesis sources were removed with `git rm` (16 files: `main.tex`, `main.pdf`, both thesis PDFs, `chapters/`, and the front-matter `.tex` files). `report/HCMUT_Master_Thesis_Template/` is now the only thesis source.

`report/latex/` still exists on purpose — it holds the **Beamer presentation**, which is a separate document:

| Kept in `report/latex/` | Why |
|---|---|
| `presentation.tex`, `presentation.pdf`, `presentation_script.md` | defense slides + 20-min speaker script |
| `references.bib` | `presentation.tex:915` does `\bibliography{references}` |
| `image/` (10 files) | `presentation.tex:27` sets `\graphicspath{{image/}{../../results/plots/}}`; `mmd_variance_change.png` and `shapedd_intuition.png` exist nowhere else |

Build entry points repointed to the HCMUT tree: `build_thesis.sh:16`, `run_all.sh:167` (thesis) / `:173` (slides, unchanged), `run_all.sh:184`, `scripts/parse_latex_log.py:28`. `build_presentation.sh` was left alone — it correctly targets `report/latex`. Verified: `pdflatex -draftmode presentation.tex` exits 0 with every image and `\input` resolved.

> Recovering anything deleted: `git show HEAD:report/latex/main.tex`, or `git checkout HEAD -- report/latex/chapters/`.

---

## 1a. What the templates actually mandate

The two forms are not uniformly prescriptive. Some pages are reproduced **verbatim with point sizes**; other things are merely named in an ordering list; and a third group isn't mentioned at all. Treating all three as equally binding was an error in an earlier draft of this review. Tiered by how literal the form is:

### Tier 1 — reproduced verbatim, so the Vietnamese wording is the spec

**Cover (`- Trang bìa và trang 1:`)** — the form prints the page with sizes attached:

| BIỂU MẪU 4 prints | Size | Thesis has |
|---|---|---|
| `ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH` | 12 | `VIETNAM NATIONAL UNIVERSITY HO CHI MINH CITY` |
| `TRƯỜNG ĐẠI HỌC BÁCH KHOA` | 13 | `HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY` |
| `--------------------` | — | *absent* |
| `HỌ VÀ TÊN HỌC VIÊN` | 16 | `LÊ PHÚC ĐỨC` (`\Large` ≈ 17) — ok |
| `TÊN ĐỀ TÀI` | 20 | English title, `\LARGE` ≈ 20 — ok |
| `Chuyên ngành:` / `Mã ngành:` | 16 | `Major:` / `Major code:` at 13pt |
| `ĐỀ ÁN TỐT NGHIỆP / LUẬN VĂN THẠC SĨ` | 20 | `MASTER'S THESIS` at **14pt** |
| `Thành phố Hồ Chí Minh, tháng … năm …` | 13 | `HO CHI MINH CITY, May 2026` — *fixed:* now driven by `\submissiondate`, size corrected 12 → 13pt |

Also `Trang bìa **và** trang 1` — the same page twice. Thesis has it once.

**Committee page (`- Trang 2:`)** — verbatim Vietnamese: `CÔNG TRÌNH ĐƯỢC HOÀN THÀNH TẠI / TRƯỜNG ĐẠI HỌC BÁCH KHOA – ĐHQG-HCM`, `Cán bộ hướng dẫn khoa học:`, `Cán bộ chấm nhận xét 1:`, `Cán bộ chấm nhận xét 2:`, each with `(Ghi rõ họ, tên, học hàm, học vị và chữ ký)`; then `Thành phần Hội đồng đánh giá … gồm:` numbered `1.`–`5.` with **no role labels**; then `CHỦ TỊCH HỘI ĐỒNG` and `TRƯỞNG KHOA`.
Thesis has `Supervisor 1/2`, `Examiner 1/2`, and labels the five as Chair / Secretary / Member / Reviewer 1 / Reviewer 2. The role labels are a harmless elaboration; the language is not.

**`PHẦN LÝ LỊCH TRÍCH NGANG`** — verbatim: `Họ và tên:`, `Ngày, tháng, năm sinh:` / `Nơi sinh:`, `Địa chỉ liên lạc:`, `QUÁ TRÌNH ĐÀO TẠO (Bắt đầu từ Đại học đến nay)`, `QUÁ TRÌNH CÔNG TÁC (Bắt đầu từ khi đi làm đến nay)`.
Thesis has `VITA` / `Full name` / `Date of birth` / `Place of birth` / `Address` / `Education Background` / `Work Experience` — and the last two are empty.

**BIỂU MẪU 5** — an entirely Vietnamese form to be reproduced and bound in. The thesis's Vietnamese task sheet is the compliant one; the extra English task sheet has no basis in either form.

### Tier 2 — named in the ordering list, in Vietnamese, but not drawn out

`Lời cám ơn` · `Tóm tắt … (tiếng Việt và tiếng Anh)` · `Lời cam đoan của tác giả` · `Mục lục` · `Toàn bộ nội dung` · `Tài liệu tham khảo` · `Phụ lục (nếu có)`.

This is a contents/ordering list. It names each item in Vietnamese but never says "the printed heading must read exactly this". Matching it is reasonable and low-risk; claiming the form *requires* it is a stretch. If you do change them:

```latex
\renewcommand{\contentsname}{MỤC LỤC}
\renewcommand{\bibname}{TÀI LIỆU THAM KHẢO}
```

### Tier 3 — not mentioned in either form

- `LIST OF TABLES`, `LIST OF FIGURES`, `LIST OF ABBREVIATIONS`, `LIST OF MATH NOTATIONS`, `LIST OF PUBLICATIONS` — the form does not list these as required front matter at all. They are additions. Normal for a thesis, but nothing in the templates governs their wording, and nothing forbids them.
- The caption words `Table` / `Figure` and the `CHAPTER n` line. **Zero mention in either form.** Pure internal consistency: Vietnamese body, English caption labels. If you want them to match the body: `\renewcommand{\tablename}{Bảng}`, `\renewcommand{\figurename}{Hình}`, `\renewcommand{\chaptername}{CHƯƠNG}`.
- **The language of the thesis body.** Neither form states it. The only language requirement anywhere in BIỂU MẪU 4 is that the abstract appear in both — `Tóm tắt … (tiếng Việt và tiếng Anh)`.

### Provenance correction (2026-07-27)

An earlier draft of this review treated the English headings as school-sanctioned because `git show 370e98f` ("Add initial LaTeX template files for HCMUT Master's…") looked like an official import. **That was wrong.** The template came from a friend enrolled in the **English-taught** programme. The author studies in Vietnamese.

So the English furniture carries no authority whatsoever — it is an artifact of borrowing an English-track template. There is no advisor question to resolve and no tier that is merely a "judgement call": all three tiers should be Vietnamese, because the thesis is Vietnamese. **All of §1a was applied on 2026-07-27** (commit "refactor: convert thesis front matter to Vietnamese").

### Practical reading (superseded — kept for the record)

Tier 1 is worth fixing regardless — those are literal mockups and the deviation is visible on the first three pages a committee member turns to. Tier 2 is cheap and makes the document coherent. **Tier 3 is a judgement call, not a compliance item** — the English furniture is what your school's own LaTeX template ships, so if an advisor has already seen a draft in this form, leave it. Worth one question to your advisor: *"the school LaTeX template prints English headings and captions but my thesis body is Vietnamese — do you want the headings Vietnamised?"*

## 2. P0 — Must fix (compliance / factual)

> **Status 2026-07-27:** 12 of 14 applied on branch `thesis-final-revisions` (commit "fix: correct P0 …"). Rebuild after the fixes: 0 errors, 0 undefined references, 0 undefined citations, 20 overfull / 15 underfull hbox (unchanged), bibliography **49 → 56** entries, **126 → 127** pages. Two items deliberately not applied — see the notes on VITA and on the Tier-1 pages below, plus the font-engine decision.

- [ ] **VITA page is blank** — *blocked: needs your personal data, cannot be filled from the repo*
  `ext_pages/vita.tex:29` (Email), `:40-54` (Education Background), `:56-67` (Work Experience)
  Both tables contain header rows only; Email is a dotfill. Confirmed empty on rendered p126.
  Supply: email to print, education history from undergraduate onward (year / degree / university), and work history (year / organisation / position). Then this is a five-minute fix.
  BIỂU MẪU 4 explicitly requires `QUÁ TRÌNH ĐÀO TẠO (Bắt đầu từ Đại học đến nay)` and `QUÁ TRÌNH CÔNG TÁC (Bắt đầu từ khi đi làm đến nay)` to be filled. This is required content left empty — a hard compliance failure.

- [x] **The four pages BIỂU MẪU 4 specifies verbatim were in English** — *applied in full*
  `ext_pages/cover.tex`, `ext_pages/committee.tex`, `ext_pages/vita.tex`, plus `ext_pages/commitment.tex` and every structural heading.
  The whole front matter is now Vietnamese, because the thesis is Vietnamese and the English template had no authority (see the provenance correction in §1a). Cover carries the Vietnamese title and the form's point sizes; the committee page is back on one page with the required `(Ghi rõ họ, tên, học hàm, học vị và chữ ký)` annotations; VITA is `PHẦN LÝ LỊCH TRÍCH NGANG` with the form's field labels.

- [ ] **PDF Vietnamese text layer is broken** — *decided 2026-07-27: not fixing, accepted risk*
  Searching the PDF for `Tóm tắt` returns **zero** hits; pure-ASCII strings like `BIBLIOGRAPHY` hit fine. `LÊ PHÚC ĐỨC` extracts as `L� PH�C C`.
  Cause: Type1 fonts without a usable ToUnicode CMap. Consequence: copy/paste is garbage, and any text-extraction plagiarism check (Turnitin/DoIT) will see mangled Vietnamese for the whole thesis.
  Fix would be `lualatex`/`xelatex` + `fontspec`. **Deliberately not applied** — different font metrics re-break every line and would shift the 127-page layout, which is not worth the risk this close to submission.
  Residual risk to confirm with the school: if the submission portal runs text extraction or a similarity check on the PDF, the Vietnamese will come through mangled. Ask before submitting; if they do extract, this becomes a blocker and the engine switch has to happen with a full visual re-check.

- [x] **DAWIDD expansion is wrong**
  `ext_pages/abbreviations.tex:41`
  Has: `Distance-Aware Windowed Drift Detection`
  Should be: `Dynamic Adapting Window Independence Drift Detection`
  Evidence: the thesis's own bib entry `hinder2020dawidd` reads *"Towards non-parametric drift detection via dynamic adapting window independence drift detection (DAWIDD)"*. The thesis contradicts its own bibliography, visibly, on p.xiv. Cheapest high-embarrassment fix — do it first.

- [x] **Six text-vs-table number mismatches** — full list in §5 below. All six are prose errors; the tables are correct.

- [x] **`mahdi2020dmddm` authors are wrong** *(printed reference — cited at `related_work.tex:97`)*
  Crossref (doi 10.1016/j.knosys.2019.105227): Osama A. Mahdi, Eric Pardede, **Nawfal Ali**, **Jinli Cao**.
  Bib has `Ali, Norazlina` and `Cao, Jian`. Both wrong.

- [x] **`yan2020acddm` author is wrong** *(printed — cited at `related_work.tex:96`)*
  Crossref (doi 10.1016/j.icte.2020.05.011): `Yan, **Myuu Myuu** Wai`.
  Bib has `Yan, Myat Myat Wai`.

- [x] **`celik2020adaptation` cites a preprint that has been published** *(printed — cited at `preliminaries.tex:41`, `related_work.tex:281`)*
  Currently `arXiv:2006.06480`. Published as **IEEE Transactions on Pattern Analysis and Machine Intelligence 43(9):3067–3078, 2021**, doi `10.1109/TPAMI.2021.3062900`. Cite the journal version.

- [x] **Friedman / Nemenyi / Critical-Difference diagram is uncited**
  `experiments.tex:126-142`
  The canonical reference for exactly that test and that figure — `demsar2006statistical` — is already in `main.bib` and never cited. Add it.

- [x] **The entire Kafka architecture section cites nothing**
  `methodology.tex:650-734`
  Eight relevant entries sit unused in `main.bib`: `kreps2011kafka`, `wang2015building`, `kleppmann2015kafka`, `goodhope2012building`, `hiraman2018apache`, `sax2018encyclopedia`, `jafarpour2019ksql`, `kafka2024documentation`.

- [x] **Figure caption contradicts the body on the same page**
  `experiments.tex:131` says the rank gap to DAWIDD is `0.142`.
  `table_statistical_tests.tex`: 3.893 − 3.464 = **0.429**, which the body at `experiments.tex:142` states correctly.

- [x] **Three conflicting submission dates**
  `ext_pages/cover.tex:41` hardcodes `HO CHI MINH CITY, January 2026`; `main.tex:32` defines `\submissiondate{May 2026}`; the task sheet gives completion `11/05/2026`. Pick one and drive the cover from the macro.

- [x] **Typo in the official task sheet**
  `ext_pages/task_sheet_VN.tex:94` — `IV. NGÀY HOÀN **THANH** NHIỆM VỤ` → `HOÀN THÀNH`.

- [x] **Task-sheet signature date precedes completion**
  `ext_pages/task_sheet_VN.tex:110` — `Tp. HCM, ngày 15 tháng 12, 2025`, but completion is `11/05/2026`. Also BIỂU MẪU 5's format is `…………………, ngày… tháng … năm 20…`.

---

## 3. P1 — Should fix (defense exposure)

- [ ] **A whole table of other people's numbers with no citations** — *decided 2026-07-27: left as is, accepted risk*
  `related_work.tex:219-250` (`tab:accuracy_results`)
  Every value is "tổng hợp từ các nghiên cứu gốc" with only a generic note — no per-row `\cite`.

  **Attempted and failed to source.** The ~20 values appear nowhere else in the repository — no notes, no script, no data file, only in this table. The D3 paper (`10.1145/3357384.3358144`) is the most likely single origin for the accuracy block, since it benchmarks D3 against exactly ADWIN / DDM / EDDM / HDDM-A / FHDDM, but ACM returns HTTP 403 so not one value could be confirmed. No citations were invented to fill the gap.

  **Author's decision: keep the table unchanged.** If asked at the defence where a specific number comes from, the honest answer is that the table aggregates figures reported in the original publications and is included as indicative context rather than as measured evidence — and that the thesis's own measurements are in Chapter 5, which is fully reproducible from `results/tables/`. Worth rehearsing that answer rather than improvising it.

- [x] **Math error in the theory chapter**
  `preliminaries.tex:148` defines `h_l(t) = max(0, 1 − |l − t| / l)`, but `:151` says it peaks at the drift moment `t = 0`.
  That formula gives `h_l(0) = 0` and peaks at `t = l`. Either write `max(0, 1 − |t|/l)` (peak at 0) or restate where the peak is. It is the central theoretical result of ShapeDD — expect it to be checked.

- [x] **The VR example undercuts the claim it supports**
  `methodology.tex:342`
  `VR_Incremental = 1.20` with `τ⁻ = 1.1` and `τ⁺ = 1.3` lands **inside the fallback band**, so VR does *not* fire the Incremental branch for that example. Yet the sentence says the result "khẳng định khả năng phân tách" of the feature.
  Fix: use an example below 1.1, or state plainly that this case falls back to the geometric rules.

- [x] **Dataset names in the prose don't match the tables** — *done: table headers relabelled GCS/GRS/Gaussian Shift; generator patched*
  Prose (`experiments.tex:31,59`): `GCS` (Gaussian Concept Stream), `GRS` (Gaussian Recurrent Stream), `Gaussian Shift (Moderate)`.
  Tables: **`Stagger`**, **`Stagger Recurrent Explicit`**, **`Gaussian Moderate`**.
  A reader cannot map your claims to your results. Rename the generated columns to match the prose.

- [x] **"14 datasets" doesn't reconcile**
  15 datasets are described (Stepping is described in the dataset section but excluded from detection, per `experiments.tex:35`); the F1 tables have 13 rows; stationary is held out for false-positive measurement. State the accounting once, explicitly, in `sec:dataset-config`.

- [x] **Caption describes data that isn't in its table**
  `experiments.tex:377` explains an `EDR` column and the value `82.5%` — `tab:cdt-comparison-by-type` has neither. Those live in `results/tables/fair_comparison.tex`, which is never `\input`. Fix the caption or add the column.

- [x] **`li2010contextual` appears to be a fabricated entry**
  Title *"Contextual multi-armed bandit algorithms for large-scale recommender systems"* with WWW 2010 pp 485–494 returns nothing in Crossref. The real Li / Chu / Langford / Schapire paper is **"A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010, pp 661–670**. Delete the entry or correct it.

- [x] **`LSTM-NDT` misattributed**
  `related_work.tex:165` — "LSTM-NDT (Neural Drift Detector dùng LSTM)`~\cite{yuan2022advances}`" cites a *survey* for a named method, and "NDT" in the literature is **Nonparametric Dynamic Thresholding** (Hundman et al. 2018, spacecraft telemetry anomaly detection), not a drift detector. Rename it or cite the primary source.

- [x] **`AEDetect` is an invented acronym**
  `related_work.tex:165` — `jaworski2020aedetect` ("Concept Drift Detection Using Autoencoders in Data Streams Processing") never names itself AEDetect.

- [x] **OCDD overstated**
  `related_work.tex:141` — "dùng one-class SVM **hoặc autoencoder**". The OCDD paper is one-class SVM only.

- [x] **ACE expansion**
  `related_work.tex:135` — "Adaptive Classifier Ensemble" → the paper's title is "Adaptive Classifier**s-**Ensemble System for Tracking Concept Drift".

- [x] **Two claims sourced to a survey that did not contain them** — *verified against the survey; one actively contradicted it*
  Fetched `hinder2024survey_partA` directly. It does **not** say D3 detects only covariate shift — it says "the used model class is crucial in terms of which drift can be detected and how much data are necessary". And it does **not** say DAWIDD has a high false alarm rate; it says DAWIDD "makes the fewest assumptions on the data or the drift… but comes at the cost of needing more data" and calls it "universally valid and surely drift-detecting".
  The DAWIDD claim was also contradicted by the thesis's own Table 5.1, where DAWIDD records **10.3 FP/run — below ShapeDD's 13.2**. Both cells rewritten to what the source actually supports.
  `related_work.tex:208` (D3 "chỉ phát hiện Covariate Shift") and `:212` (DAWIDD "Tỷ lệ báo động giả khá cao"), both attributed to `hinder2024survey_partA`. Verify both are actually in that survey. Sourcing DAWIDD's weakness to a survey co-authored by DAWIDD's own author is an easy question to get asked.

- [x] **`VR` missing from the abbreviations list**
  `ext_pages/abbreviations.tex` — Variance Ratio is the thesis's own added feature and the only one absent.

- [x] **Footnote marker/text mismatch**
  `ext_pages/publication.tex:11` `\footnotemark[7]` vs `:27` `\footnotetext[8]` → renders as `**` in the body and `††` in the footnote. Confirmed on p94.
  Same file: a Vietnamese sentence (`:15`) on an otherwise English page, and a CRediT boilerplate footnote when there are no publications. Drop the footnote.

- [x] **Uncited quantitative claims in the opening**
  `introduction.tex:4-8` — "tỷ lệ cảnh báo sai tăng gấp ba lần" and the steel-factory anecdote. Specific numbers, no source. Several suitable motivation entries sit unused (`wuest2016machine`, `wu2021dependable`, `chui2021state`, `chui2022state`, `biegel2022combining`).

- [x] **ADWIN memory bound uncited**
  `preliminaries.tex:242` — `O(log W)` claim needs `bifet2007learning`.

- [x] **Unsupported bandwidth claim**
  `methodology.tex:10` — IDW-MMD "giảm độ nhạy với bandwidth". No Ch.5 experiment measures bandwidth sensitivity. Drop it or reframe as design intent rather than result.

---

## 4. P2 — Nice to have

- [x] **Straight double quotes render as `”x”` instead of `“x”`** — 10 places: `experiments.tex:191`; `methodology.tex:622,732`; `preliminaries.tex:205`; `related_work.tex:24,125,129,139,140,161`. Plus raw Unicode curly quotes at `preliminaries.tex:170`. Use `` ``…'' ``.
- [x] `methodology.tex:393` — `\begin{tabular}{llll}` with a 3-cell header row gives `tab:se-cdt-params` a spurious empty 4th column.
- [x] `methodology.tex:122` and `:160` — self-referential `\ref{sec:idw-mmd}` from inside `sec:idw-mmd` (label at `:115`).
- [x] **Window model inconsistent** — `preliminaries.tex:251-256` describes ShapeDD as a symmetric double window `2l₁` (both halves `l₁`); methodology and appendix use asymmetric `l₁ = 50, l₂ = 150`. Reconcile in the theory chapter.
- [x] **Four unreconciled window sizes** — classification `W = 50` (`experiments.tex:291`), Kafka `W = 250` (`:481`), H0 `W = 300` (`:219`), appendix A.4 buffer 750 / chunk 150 vs `methodology.tex:688` "Circular Buffer (ví dụ: 1000 mẫu)". Add a `W` column to `tab:eval-conventions`, which currently reconciles only `δ` and cooldown.
- [x] **Parameter tables disagree** — appendix A.2 has `PPR = 0.20` and `DPAR = 0.60`, absent from `tab:se-cdt-params`; that table lists `τ_WR / τ_SNR / τ_CV` as "self-calibrated" with no static defaults, while the conclusion quotes them as hard numbers (`WR < 0.15`, `SNR > 2.0`, `CV < 0.3`). Make one table authoritative and cross-reference from the other.
- [x] **"Growth process" is undefined** — `conclusion.tex:38` and `:46` use it as a CDT-MSW mechanism, and `:46` points at `sec:shaped-cdt`. Neither `related_work.tex` §CDT-MSW (which calls it "phương sai độ chính xác trên cửa sổ phụ `W_R`") nor `methodology.tex` defines the term.
- [x] **Notations list incomplete** — *done: symbols added and descriptions translated* — `ext_pages/notations.tex` is missing `σ(t)` (the central signal), `h_l(t)`, `VR`, `τ_VR^±`, `M`, `n_s`, `σ_g`, `δ`, `N_perm`.
- [x] **Same quantity, two symbols** — *done* — `table_se_cdt_performance_by_type.tex` uses `\tau_{\text{match}} = 0.15`; `table_cdt_msw_vs_se_cdt_by_drift_type.tex` uses `\mu = 0.15`.
- [x] `table_statistical_tests.tex` — *done, now shown as 2--4.* Was assigning Overall Rank 2 / 3 / 4 assigned to three methods with an identical Average Rank of 3.893. Show as tied (2–4).
- [x] **Cover details vs BIỂU MẪU 4** — `ext_pages/cover.tex:37` sets `MASTER'S THESIS` at 14pt (template says 20); `:41` sets city/date at 12pt (template says 13); missing the `--------------------` rule under `TRƯỜNG ĐẠI HỌC BÁCH KHOA`; template says "Trang bìa **và trang 1**" so the title page should appear twice.
- [x] `ext_pages/committee.tex` — *done, back on one page.* Was spilling onto 2 pages because of `\vspace{10cm}`; BIỂU MẪU 4 puts it on page 2 alone.
- [x] **Two task sheets** — *done: English copy removed (task_sheet_VN already prints both titles)* — EN then VN. Template requires only BIỂU MẪU 5 (Vietnamese). EN numbers sections I–VI, VN numbers I–V.
- [x] **Abstract order** — *done: Vietnamese first* — BIỂU MẪU 4 lists "Tóm tắt … (tiếng Việt và tiếng Anh)"; `main.tex:45-46` has English first. Swap.
- [x] **Front-matter heading sizes** — *done: unified* — 24pt (abbreviations, notations, publications) vs `\Large` ≈17pt (ack, abstract, commitment). Unify.
- [ ] **Bib hygiene** — prune or wire up the 50 uncited entries; drop the `scott2015multivariate` / `scott1992multivariate` duplicate (same book, two editions, both unused); `tripathi2021ensuring` `pages={22}` → article number `576892`; `haug2024benchmark` key says 2024 for Lukats et al. 2025.
- [x] **Dead files** — *done for the three unused `.tex` files and `build_thesis.sh`.*
  **`results/tables/fair_comparison.tex` deliberately kept.** It is never `\input`, so it looks dead — but it is a *generated results artifact*, and since `results/raw/` no longer exists it is the only surviving record of those figures (CDT-MSW CAT 87.0 / SUB 86.7 / Recall 82.5). Deleting it would destroy data that cannot be regenerated. Leave it in place. — `chapters/discussion.tex`, `chapters/results.tex` (commented out at `main.tex:80-81`), `ext_pages/ack.tex` (unused, empty), `results/tables/fair_comparison.tex` (never `\input`).
- [x] **Retire the duplicate thesis tree** — done 2026-07-27, see §1. `report/latex/` now holds only the presentation.
- [x] `build_thesis.sh` removed; `build_template_thesis.sh` kept and docker-compose.yml repointed. Both had built the same document from the same directory, differing only in output filename (`…ThesisReport` vs `…ThesisReport_HCMUT`). Pick one and delete the other.

---

## 5. Reference — text vs generated tables

The tables are auto-generated. **Fix the prose, not the tables.** All seven corrected 2026-07-27 and confirmed present in the rebuilt PDF.

| Location | Prose said | Table says | Status |
|---|---|---|---|
| `experiments.tex:117` | DAWIDD F1 highest at **0.531** | `table_I`: DAWIDD **0.532**, ShapeDD-IDW 0.531 — so they do not tie numerically | fixed; paragraph also reworded to state the 0.001 gap and redirect the claim to runtime + classification |
| `experiments.tex:119` | ShapeDD FP **13.1** | `table_I`: **13.2** | fixed |
| `experiments.tex:185` | ShapeDD FP **13.1** (again) | `table_I`: **13.2** | fixed |
| `experiments.tex:131` | CD gap to DAWIDD **0.142** | `table_statistical_tests`: 3.893 − 3.464 = **0.429** | fixed |
| `experiments.tex:181` | Random Mild, MMD **0.563** | `table_II_part1`: **0.566** | fixed |
| `experiments.tex:181` | Gaussian Gradual, MMD **0.459** | `table_II_part1`: **0.462** | fixed |
| `experiments.tex:183` | RBF Blips, MMD **0.742** | `table_II_part3`: **0.747** — 0.742 is MMD's *Stagger* value, i.e. a column slip | fixed |

> Verification note: `0.742` and `0.459` still each appear once in the PDF — both in the legitimate `Stagger` row of `table_II_part3` (MMD 0.742, KS 0.459), not in prose. Don't "fix" those.

Values confirmed correct and needing no change: `9.6` FP/run, `0.491` ShapeDD F1, `10.5` MMD FP, `27.2` KS FP, Hyperplane `0.179`, LED Abrupt `0.148`, Standard SEA `0.118`, D3 RBF Blips `1.000`, `7.18×`/`≈7.2×` speedup, `0.429` and `0.714` rank gaps at `:142`, all Type-I error figures at `:230-232`, CDT-MSW `96.9%`/`74.0%`/`83.9%`, and all prequential accuracy figures.

## 6. Reference — citation errors

| Entry | Printed? | Problem | Source of truth | Status |
|---|---|---|---|---|
| `mahdi2020dmddm` | **yes** | `Ali, Norazlina` → `Ali, Nawfal`; `Cao, Jian` → `Cao, Jinli` | Crossref `10.1016/j.knosys.2019.105227` | fixed |
| `yan2020acddm` | **yes** | `Myat Myat Wai` → `Myuu Myuu Wai` | Crossref `10.1016/j.icte.2020.05.011` | fixed |
| `celik2020adaptation` | **yes** | preprint cited; use IEEE TPAMI 43(9):3067–3078, 2021 | Crossref `10.1109/TPAMI.2021.3062900` | fixed (`@misc` → `@article`) |
| `tripathi2021ensuring` | **yes** | `pages={22}` → article no. `576892` | Frontiers in AI 4 | fixed |
| `schrab2023mmdagg` | no | `Albert, Mika` → `Albert, Mélisande`; **`Guedj, Benjamin` missing**; pages `1--72` → `1--81` | JMLR 24(194) |
| `li2010contextual` | no | title and pages do not correspond to any real paper | Crossref returns no match |
| `haug2024benchmark` | no | key says 2024; work is Lukats et al. **2025** (metadata otherwise correct) | Crossref `10.1007/s41060-024-00620-y` |
| `scott2015multivariate` + `scott1992multivariate` | no | same book, two editions, both unused | — |

### Entries that exist but are never cited, where the text needs them

| Needed at | Entry sitting unused | Status |
|---|---|---|
| `experiments.tex:126` — Friedman/Nemenyi + CD diagram | `demsar2006statistical` | **cited** |
| `methodology.tex:653` — Kafka architecture | `kreps2011kafka`, `kafka2024documentation`, `wang2015building`, `kleppmann2015kafka`, `goodhope2012building`, `hiraman2018apache` | **cited** — a short paragraph now justifies *why* Kafka (durable partitioned commit log, independent consumers at different cadences) rather than just naming it |
| — | `sax2018encyclopedia`, `jafarpour2019ksql` | still unused; KSQL and the encyclopedia entry aren't needed by the current design |
| `experiments.tex:401` — LogisticRegression + StandardScaler | `pedregosa2011scikit` |
| Gaussian smoothing, FWHM, peak detection | `savitzky1964smoothing`, `canny1986computational`, `hamming1989digital` |
| Permutation-test theory | `neyman1933problem`, `ramdas2023permutation` |
| The survey series the thesis descends from | `hinder2024survey_partB` |
| `introduction.tex` industry motivation | `wuest2016machine`, `wu2021dependable`, `chui2021state`, `chui2022state`, `biegel2022combining` |

---

## 7. Methodology assessment

**Verdict: the update is sound. Do not touch the method.**

What works:
- **VR is properly grounded.** The Law of Total Variance is the right lens for separating a probabilistic mixture (Gradual → between-concept variance inflates) from a continuous mean shift (Incremental → variance roughly constant). It targets a *measured* failure rather than a hypothetical one, and it is reported with its limits intact.
- **Concept Memory and the one-directional self-calibration are modest and defensible.** Neither over-claims; the self-calibration explicitly only loosens thresholds, preserving behaviour on clean streams.
- **The candour is a genuine strength.** Type-I 0.140 on AR(1), 4846 FP, SUB 55.3%, and the CDT-MSW comparison explicitly labelled non-equivalent. Keep every bit of this — it reads as rigour, not weakness. Do not soften it under pressure.

### Six things to rehearse

1. **Detection is a tie, not a win.** F1 0.531 vs DAWIDD 0.532, and DAWIDD leads on average rank (3.464 vs 3.893). The actual contribution is the **7.18× speedup plus the type-classification layer**. The abstract and conclusion already frame it that way; `experiments.tex:117` reads defensively by comparison. Align it — lead with speed and classification.

2. **Don't lean on the Nemenyi tie as evidence.** CD = 2.806 across 8 methods × 14 datasets can barely separate anything. "Not statistically distinguishable" is weak evidence, not evidence of parity. Saying this yourself turns a vulnerability into a display of statistical literacy.

3. **Composite Type-I error reaches 0.140 (2.8α)** while a core selling point is "a p-value with statistical grounding instead of a manual threshold". Bonferroni is demonstrably insufficient here. The correct answer — peak selection plus dependent sliding windows makes this a *selective inference* problem, so per-test correction cannot control the pipeline-level rate — is already half-written at `experiments.tex:232`. Sharpen it into one clean sentence.

4. **The 4846 → 657 FP swing** is presented as a configuration choice. The first question will be "which configuration produced 80.1%". Restate in the table note that CAT/SUB are measured in oracle mode and are therefore identical across both configurations.

5. **Concept Memory absorbs 57% of Gradual events**, which makes part of the classifier's error *architectural* — memory runs after threshold classification and can relabel. A Concept-Memory-off ablation is cheap and would convert this from a weakness into an analysed design trade-off. **Optional** — this is the only item requiring new computation, and the methodology is frozen. Skip it if time is short; just be ready to describe the mechanism verbally.

6. **All detection evidence is synthetic plus one semi-synthetic dataset**, while Ch.1 motivates the work on industrial manufacturing quality control. Already acknowledged in the limitations. Keep Ch.1's promises proportional so the gap reads as scoping, not oversight.

---

## 8. Related drift in the defense-prep docs

Found while checking repo conventions — not part of the thesis PDF, but it affects the defense.

Current `results/tables/table_III_runtime_stats.tex`:

| Method | Runtime | Throughput | Speedup |
|---|---|---|---|
| ShapeDD-IDW | **0.59 s** | 16,858 | **7.18×** |
| ShapeDD (gốc) | **4.26 s** | 2,349 | 1.00× |
| DAWIDD | **5.66 s** | 1,767 | 0.75× |

The prep docs still quote the previous run:

- `DEFENSE_PREP_VN.md:117` — "0.70s / 5.03s / 6.79s (7.17× / …)"
- `DEFENSE_PREP_VN.md:108` — "nhanh hơn ~7×–9.7×"
- `THESIS_GUIDE.md:1326` — "DAWIDD | 6.79 | 1,473 | 0.74×"
- `THESIS_GUIDE.md:1774` — "Runtime beats DAWIDD (~9.7×: 6.79s vs 0.70s)"

All three absolute runtimes and both throughput figures are stale. The `~9.7×` ratio survives only coincidentally (5.66 / 0.59 ≈ 9.6×). **Rehearsing from these documents means quoting seconds that don't match the submitted PDF.** Refresh them from the table.

Second issue: **no root document mentions `HCMUT_Master_Thesis_Template`.** The prep material was written against the tree that is not the deliverable. After the §1 cleanup, two of those references now point at **files that no longer exist**:

- `DEFENSE_PREP_VN.md:5` — "toàn bộ 6 chương trong `report/latex/chapters/`" (deleted)
- `THESIS_GUIDE.md:1972` — "Companion to thesis PDF `report/latex/2370116_LePhucDuc_ThesisReport.pdf` (102 pages…)" (deleted; and the deliverable is 126 pages, not 102)
- `README_DEFENSE.md:260` — the directory tree diagram still shows `report/latex/  ← LaTeX thesis source`

Repoint all three to `report/HCMUT_Master_Thesis_Template/`. Left untouched here because these are your rehearsal notes, not the submission — but fix them before you rehearse, together with the stale runtimes above.

---

## 9. Appendix — how to re-verify

All commands from the repo root, Git Bash.

**Rebuild and collect warnings** (writes `main.pdf`; `git checkout` it afterwards if you only wanted the log):

```bash
cd report/HCMUT_Master_Thesis_Template
pdflatex -interaction=nonstopmode -file-line-error main.tex > /tmp/pass1.txt 2>&1
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex > /tmp/pass2.txt 2>&1
pdflatex -interaction=nonstopmode -file-line-error main.tex > /tmp/pass3.txt 2>&1

grep -nE "^!" /tmp/pass3.txt                                  # errors
grep -inE "undefined (reference|citation)" /tmp/pass3.txt      # unresolved
grep -c "Overfull" /tmp/pass3.txt
grep -c "bibitem" main.bbl                                     # printed references
```

**Cited vs defined bib keys** — count `@`-entries in `main.bib`, collect `\cite*{...}` keys across `chapters/`, `ext_pages/`, `main.tex` (skipping `%`-commented lines), then diff the two sets both ways. Note when writing such a script in Python that a literal backslash in the regex must be built as `chr(92)*2 + 'cite...'`; a raw string `r'\cite'` raises `bad escape \c`.

**Label / ref integrity** — same idea over `\label{...}` vs `\ref{...}`/`\eqref{...}`, including `results/tables/*.tex` (some labels live inside the generated table files).

**Read the official templates:**

```bash
antiword -m UTF-8 -w 110 "BIỂU MẪU 4"*.doc     # older .doc binary format
python -c "import zipfile,re,html; z=zipfile.ZipFile('BIEU_MAU_5.docx'); \
  x=z.read('word/document.xml').decode('utf8'); \
  x=x.replace('</w:p>','\n').replace('</w:tc>',' | '); \
  print(html.unescape(re.sub(r'<[^>]+>','',x)))"
```

Both filenames are stored in NFD (decomposed) Unicode — `os.listdir` and glob work, but a hardcoded NFC literal will not match. Copy them to ASCII names first.

**Render a page to check layout** (requires `pymupdf`):

```bash
python -c "import fitz; d=fitz.open('report/HCMUT_Master_Thesis_Template/main.pdf'); \
  d[110].get_pixmap(dpi=110).save('/tmp/p111.png')"
```

> Do this before reporting any table as broken. `pdftotext -layout` misrenders nested `p{}` tabulars and produced two false positives in this review.

**Verify a citation:**

```bash
curl -s "https://api.crossref.org/works/10.1016/j.knosys.2019.105227" | python -m json.tool | head -40
curl -s "https://api.crossref.org/works?query.bibliographic=<title>&rows=3"
```
