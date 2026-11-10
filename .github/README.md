# Quy trinh nhanh & CI/CD

Nhanh: `dev` (phat trien) -> `main` (on dinh). Chi merge vao `main` tu `dev`, tu dong khi CI xanh (`promote.yml`).

## Thiet lap thu cong tren GitHub (mot lan)
1. **Settings -> Actions -> General -> Workflow permissions**: chon *Read and write permissions* va tick
   *Allow GitHub Actions to create and approve pull requests*.
2. **Settings -> General -> Pull Requests**: bat *Allow auto-merge*.
3. **Settings -> Branches** (hoac Rulesets), tao rule cho `main`:
   - Require a pull request before merging
   - Require status checks to pass: `test (py3.10)`, `test (py3.11)`, `test (py3.12)`
   - Block force pushes, khong cho xoa nhanh
4. Rule cho `dev`: Require status checks (cac check `test ...` nhu tren), block force pushes.
5. Dat `dev` lam nhanh mac dinh neu muon PR mac dinh nham vao `dev`.

Luu y: PR do `GITHUB_TOKEN` tao khong tu kich hoat workflow `pull_request`; khong dat `enforce-source`
lam required check cho PR tu dong. Cac check `test` da chay tren commit cua `dev` nen van thoa dieu kien merge.
