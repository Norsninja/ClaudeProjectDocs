# Git Commit Plan for Auth Implementation

## Repository Structure
- **Git Root**: `/e/Projects/contexthub-saas/handoffkit/`
- **Parent Directory**: Preserves Claude conversations (not in git)
- **Working Branch**: master (should consider feature branches)

## Current Changes Overview
22 files modified/created during auth implementation

## Recommended Git Flow

### 1. Immediate Actions
```bash
# Navigate to git repository
cd /e/Projects/contexthub-saas/handoffkit

# Review all changes
git status
git diff

# Stage auth-related changes in logical groups
```

### 2. Commit Strategy (Logical Groupings)

#### Commit 1: Core Auth Infrastructure
```bash
git add apps/web/src/lib/auth/
git add apps/web/src/lib/supabase/server.ts
git commit -m "feat: Add auth infrastructure with Zod validation and error handling

- Add validation schemas for sign-in/sign-up
- Implement error mapping for user-friendly messages
- Create workspace bootstrap with retry logic
- Update server.ts for Next.js 15 async cookies"
```

#### Commit 2: Auth Pages & Components
```bash
git add apps/web/src/app/\(auth\)/
git add apps/web/src/components/setup-incomplete.tsx
git add apps/web/src/app/auth/
git commit -m "feat: Implement auth pages and flows

- Add sign-in/sign-up pages with server actions
- Create check-email and error pages
- Implement setup-incomplete interstitial
- Add email confirmation route with workspace bootstrap"
```

#### Commit 3: Projects Page Integration
```bash
git add apps/web/src/app/\(app\)/projects/
git commit -m "feat: Integrate auth with projects page

- Add workspace detection in projects page
- Implement retry workspace setup action
- Handle setup incomplete state"
```

#### Commit 4: RLS Fix
```bash
git add supabase/migrations/0002_fix_rls_recursion.sql
git commit -m "fix: Resolve RLS infinite recursion bug

- Simplify workspace_members SELECT policy
- Break circular dependency between policies
- Fixes workspace query failures"
```

#### Commit 5: Documentation
```bash
git add PRODUCTION_CHECKLIST.md
git add AUTH_IMPLEMENTATION_STATUS.md
git add RLS_RECURSION_BUG.md
git add GIT_COMMIT_PLAN.md
git commit -m "docs: Add auth implementation documentation

- Production checklist with email templates
- Implementation status report
- RLS recursion bug analysis
- Git workflow documentation"
```

#### Commit 6: Dependencies
```bash
git add apps/web/package.json apps/web/package-lock.json
git commit -m "chore: Add auth dependencies

- Add zod for validation
- Add @radix-ui/react-label for forms"
```

## Best Practices Going Forward

### Branch Strategy
```bash
# For future features
git checkout -b feat/projects-crud
git checkout -b feat/cells-management
```

### Commit Message Format
```
type(scope): description

- Detail 1
- Detail 2

Fixes #issue
```

Types: feat, fix, docs, chore, refactor, test

### Regular Commits
- Commit at each milestone completion
- Don't let 20+ files accumulate
- Use atomic commits (one logical change per commit)

### Before Pushing
1. Run tests (when implemented)
2. Check for console.logs to remove
3. Verify no secrets in code
4. Update documentation if needed

## Current Status Check
- [ ] Review all changes with `git diff`
- [ ] Remove debug console.logs
- [ ] Stage files in logical groups
- [ ] Write clear commit messages
- [ ] Consider creating release tag: `v0.1.0-auth`

## Remote Repository
- Ensure `.gitignore` includes:
  - `.env.local`
  - `node_modules/`
  - `.next/`
  - Claude conversation directories

## Protection Rules
- Never commit from root `/contexthub-saas/`
- Always work within `/handoffkit/`
- Keep Claude conversations out of git