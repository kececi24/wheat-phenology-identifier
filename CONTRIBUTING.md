# Contributing to AI Connect

This file holds the naming rules for **branches** and **commit messages**, and explains the **CI checks** every pull request must pass. Every member follows them for every change.

---

## 1. Branch names

### Pattern

```
<type>/<short-description>
```

The existing branches in the repository (`docs/charter`, `docs/project_plan`) follow this pattern.

### Rules

- Lowercase and **English**; no non-ASCII characters.
- Words are separated by **hyphens (`-`)**, as in the course guide (`docs/project-plan`). Underscores are not used in new branches.
- `<type>` is one of the commit types in section 2.2: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `style`.
- `<short-description>` is 2–4 words and says what the work is, not who does it.
- One branch is for one issue. It is deleted after its PR is merged.
- Course document branches are kept until their revision PR is merged.

### Examples

| Good | Bad | Why it is bad |
|---|---|---|
| `feat/backend-skeleton` | `ali` | Says who, not what |
| `chore/docker-compose-core` | `new-branch`, `test2` | Says nothing |
| `feat/account-jwt-login` | `Feat/Backend_Skeleton` | Uppercase and underscore |
| `fix/feed-pagination-order` | `feat/giriş-sayfası` | Non-ASCII characters, not English |

### Course document branches (fixed)

These names come from the course guide and are not changed.

| Deliverable | Branch | File | Release name |
|---|---|---|---|
| Requirements Spec | `docs/requirements-spec` | `docs/RequirementsSpec.md` | `requirements-spec-v1.0` |
| Requirements Analysis | `docs/requirements-analysis` | `docs/RequirementsAnalysis.md` | `requirements-analysis-v1.0` |
| Design Spec | `docs/design-spec` | `docs/DesignSpec.md` | `design-spec-v1.0` |
| Test Spec | `docs/test-spec` | `docs/TestSpec.md` | `test-spec-v1.0` |

A document is revised in the same file; no versioned copies such as `RequirementsSpec_v1.1.md`.

---

## 2. Commit messages

### 2.1 Pattern

The repository uses **Conventional Commits**:

```
<type>(<scope>): <summary>
```

Commits already in the repository:

```
docs(charter): add charter v1.0 and v1.1
docs(readme): add README
chore(git): add .gitignore
docs(plan): add project_plan
```

### Rules

- Written in **English**.
- `type` and `scope` are lowercase.
- The summary is in the **imperative mood**: `add`, `fix`, `update`, `remove`, `rename` (not `added`, `fixes`, `adding`). It completes the sentence "If applied, this commit will ___".
- The summary starts with a lowercase letter, has no trailing period and is at most 72 characters.
- **One commit = one logical change.** "Added login and changed the navbar color" is two commits.
- When the reason is not obvious, add a body after a blank line:

```bash
git commit -m "fix(feed): order posts by created_at descending" \
           -m "The feed was sorted by id, which breaks when posts are imported out of order."
```

- **Pull request titles use the same format** (PR #1: `docs(plan): add project_plan`).

### 2.2 Types

| Type | When | Example |
|---|---|---|
| `feat` | A new feature | `feat(account): add register endpoint` |
| `fix` | A bug fix | `fix(post): reject uploads larger than 20 MB` |
| `docs` | Documentation only | `docs(req-spec): add user stories for groups` |
| `test` | Adding or fixing tests only | `test(account): cover expired refresh token` |
| `refactor` | Code change that does not change behavior | `refactor(feed): extract cursor pagination helper` |
| `chore` | Configuration, dependencies, repository housekeeping | `chore(docker): add redis service` |
| `ci` | GitHub Actions workflows | `ci: run backend tests on pull requests` |
| `style` | Formatting only, no logic change | `style(frontend): apply prettier` |

### 2.3 Scopes

The scope says **where** the change is. Leave it out when the change does not belong to a single area (`ci: ...`).

| Group | Scopes |
|---|---|
| Backend modules | `account`, `post`, `feed`, `article`, `issue`, `community`, `search`, `recommendation`, `ai`, `notification` |
| Infrastructure | `backend`, `frontend`, `docker`, `db`, `ci`, `deps`, `git` |
| Design | `design`, `ui` |
| Documents | `charter`, `plan`, `req-spec`, `req-analysis`, `design-spec`, `test-spec`, `readme`, `contributing` |

### 2.4 Examples

| Good | Bad |
|---|---|
| `feat(account): add login endpoint with jwt` | `login` |
| `fix(community): reject posts in a pending community` | `bug fix` |
| `chore(docker): add minio service and bucket init` | `docker stuff` |
| `docs(req-spec): add acceptance criteria for issues` | `update` |
| `test(post): add tests for delete permission` | `asdfgh`, `final`, `final2`, `last version` |

---

## 3. CI checks

Every pull request to `main`, and every push to `main`, runs the workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). All five jobs are required status checks: a PR with a red check cannot be merged.

| Job | What it checks | Run it locally |
|---|---|---|
| `backend-lint` | `ruff check .` and `ruff format --check .` in `backend/` | `make lint` and `make format-check` (`make format` fixes formatting) |
| `backend-test` | `alembic upgrade head`, then `pytest` against PostgreSQL 16 with a **70% coverage gate** on `app/` | `make migrate` and `make test-backend-cov` |
| `backend-audit` | `pip-audit` on `requirements.txt` and `requirements-dev.txt` | `make audit` |
| `frontend` | `npm ci`, `lint`, `format:check`, `typecheck`, `test` (Vitest), `build`, `npm audit --audit-level=high` | `make lint`, `make format-check`, `make typecheck`, `make test-frontend`, `make build-frontend`, `make audit` |
| `docker` | `docker compose config` and a build of the backend and frontend images | `make config` and `make build` |

`make ci` runs all of them in the same order as the pipeline. Run it before you push.

- Backend checks run inside the backend container. Start it with `make up-build` (rebuild after `requirements*.txt` changes).
- Frontend checks run on your machine, in `frontend/`. Install Node from [`frontend/.nvmrc`](frontend/.nvmrc) (for example `nvm use`) and run `npm ci` once.

### When a check fails

1. Open the failed job in the **Checks** tab of the PR and read the log of the red step.
2. Run the same command locally (table above) and fix the problem.
3. Push the fix to the **same branch**. CI runs again and cancels the previous run. Do not open a new PR.

Common fixes:

- **Formatting:** `make format`, then commit the result.
- **Coverage below 70%:** add tests for the code you changed. The `term-missing` table in the log lists the uncovered lines, and the full report is attached to the run as the `backend-coverage` artifact.
- **Audit:** upgrade the package to the fixed version the audit names. If there is no fix yet, say so in the PR and ask the reviewer.
- **A check you did not touch fails on `main` as well:** tell the team before merging anything else.

### Optional: pre-commit hooks

[`.pre-commit-config.yaml`](.pre-commit-config.yaml) runs the fast checks (ruff, ruff format, prettier, merge-conflict markers, private keys, large files) before each commit. It is **optional**: CI is the gate, so nobody is blocked if they do not install it.

```bash
pip install pre-commit
cd frontend && npm ci && cd ..   # prettier runs from frontend/node_modules
pre-commit install
```

- Run all hooks once with `pre-commit run --all-files`. Skip them for one commit with `git commit --no-verify`; CI still runs.
- The ruff version in the config must equal `ruff` in `backend/requirements-dev.txt`. Do not run `pre-commit autoupdate` without checking that.
- Hooks that rewrite files skip `docs/`, which holds the course deliverables.
