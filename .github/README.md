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
   - Require status checks: `test (py3.10)`, `test (py3.11)`, `test (py3.12)`
   - Block force pushes, khong cho xoa nhanh
4. Rule cho `dev`: Require status checks (cac check `test ...`), block force pushes, khong cho xoa nhanh.
5. Dat `dev` lam nhanh mac dinh.

Luu y: PR do `GITHUB_TOKEN` tao khong tu kich hoat workflow `pull_request`; khong dat `enforce-source`
lam required check. Cac check `test` da chay tren commit cua `dev` nen van thoa dieu kien merge.
