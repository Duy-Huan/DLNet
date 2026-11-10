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

Luu y: PR do `GITHUB_TOKEN` tao khong tu kich hoat workflow `pull_request`; khong dat `enforce-source`
lam required check. Cac check `test` da chay tren commit cua `dev` nen van thoa dieu kien merge.

## Cac job CI
| Job | Chay gi | Muc dich |
|---|---|---|
| `lint` | `ruff check`, `ruff format --check` | style, loi tinh |
| `test (py3.10-3.12)` | `pytest -m "not e2e"` voi torch CPU; **fail neu co test bi skip** | unit + regression tren 3 ban Python |
| `regression` | `pytest -m regression` | ten rieng de thay ngay khi mot loi cu quay lai |
| `e2e` | `pytest -m e2e` | CLI that -> goi xuat ra -> `pytest` va `train.py` trong goi (tabular + HSI) |

Moi tuan (thu Hai 03:17 UTC) CI tu chay lai tren nhanh mac dinh de bat loi do nang cap torch/scikit-learn.
`promote.yml` goi ca `ci.yml`, nen `dev -> main` chi xay ra khi moi job tren xanh.
