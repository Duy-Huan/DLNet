# Quy trinh nhanh & CI/CD

Chi co 2 nhanh: `dev` (phat trien, push truc tiep) va `main` (on dinh, khong push truc tiep).
Khong tao them nhanh khac. Khi push vao `dev` va CI xanh, `promote.yml` tu tao PR `dev -> main` va merge.

## Thiet lap thu cong tren GitHub (mot lan)
1. **Settings -> Actions -> General -> Workflow permissions**: *Read and write permissions* va tick
   *Allow GitHub Actions to create and approve pull requests*.
2. **Settings -> General -> Pull Requests**: bat *Allow auto-merge*; TAT *Automatically delete head branches*
   (neu bat, `dev` co the bi xoa sau khi merge).
3. **Settings -> Branches** (hoac Rulesets), rule cho `main`:
   - Require a pull request before merging
   - Require status checks: `lint`, `test (py3.10)`, `test (py3.11)`, `test (py3.12)`, `regression`, `e2e`
   - Block force pushes, khong cho xoa nhanh
4. Rule cho `dev`: Require status checks (`lint`, `test ...`, `regression`, `e2e`), block force pushes, khong cho xoa nhanh.
5. Dat `dev` lam nhanh mac dinh.
6. **Settings -> Rules -> Rulesets -> New tag ruleset**: pattern `v*`, chan xoa va chan cap nhat tag (tag phat hanh la bat bien). Workflow `release.yml` van tao tag duoc (khong dua bot vao danh sach chan tao).
7. Release dung `GITHUB_TOKEN` voi `contents: write` (da co trong Workflow permissions o muc 1).

Luu y: CI khong co trigger `pull_request` (PR do `GITHUB_TOKEN` tao chua tung chay duoc job nao). Cac check
(`lint`, `test`, `regression`, `e2e`) chay tren push vao `dev` va gan vao dung SHA, nen van thoa dieu kien merge vao `main`.

## Cac job CI
| Job | Chay gi | Muc dich |
|---|---|---|
| `lint` | `ruff check`, `ruff format --check` | style, loi tinh |
| `test (py3.10-3.12)` | `pytest -m "not e2e"` voi torch CPU; **fail neu co test bi skip** | unit + regression tren 3 ban Python |
| `regression` | `pytest -m regression` | ten rieng de thay ngay khi mot loi cu quay lai |
| `e2e` | `pytest -m e2e` | CLI that -> goi xuat ra -> `pytest` va `train.py` trong goi (tabular + HSI) |

Moi tuan (thu Hai 03:17 UTC) CI tu chay lai tren nhanh mac dinh de bat loi do nang cap torch/scikit-learn.
`promote.yml` goi ca `ci.yml`, nen `dev -> main` chi xay ra khi moi job tren xanh.

## Release
`promote.yml` goi `release.yml` ngay sau khi merge dev vao main. `release.yml` doc `version` trong `pyproject.toml`;
neu chua co tag `vX.Y.Z` thi kiem tra (`scripts/release.py check`), build wheel, thu cai, roi tao tag va GitHub Release
voi ghi chu lay tu `CHANGELOG.md`. Chay tay: **Actions -> Release -> Run workflow** (dung khi auto-merge bi tre).
Quy trinh day du: `agent.md` §11.2 va `skills/release/SKILL.md`.
