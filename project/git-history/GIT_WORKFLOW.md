# HandoffKit Git Workflow

## Strategy: Linear Development with Milestone Tags

### Current Approach
- **Branch**: Stay on `master` 
- **History**: Keep linear and clean
- **Milestones**: Use annotated tags
- **Context**: Preserve Claude continuity

### Milestone Tags
- ✅ `v0.1.0-auth` - Authentication complete
- 🎯 `v0.2.0-projects-crud` - Next milestone
- 📋 `v0.3.0-cells` - Future milestone

### Commit Conventions
```
type(scope): description

Types: feat, fix, docs, refactor, chore, perf, test
```

### Pre-Commit Checklist
```bash
# 1. Check status
git status
git diff --stat

# 2. Scan for secrets (should return nothing)
git grep -nE "(SUPABASE|KEY|SECRET|TOKEN)" -- :!**/*.md

# 3. Check for debug logs
git grep "console\." -- apps/web/src

# 4. Verify .env ignored
git check-ignore -v apps/web/.env.local
```

### Pre-Tag Testing
1. **Clean boot**: `npm run dev`
2. **Auth flow**: Sign up → Workspace created → Projects load
3. **No console errors**
4. **Migrations applied**

### Tagging Process
```bash
# Create annotated tag
git tag -a v0.2.0-projects-crud -m "Projects CRUD foundation"

# Verify tags
git tag -l

# View tag details
git show v0.2.0-projects-crud
```

### Rollback if Needed
```bash
# Safe while local-only
git reset --hard v0.1.0-auth

# Or checkout tag to inspect
git checkout v0.1.0-auth
```

### Session Tags (Optional)
For major work sessions:
```bash
git tag session-2025-09-10-pm
```

### When to Add Branches
Introduce feature branches when:
- Adding GitHub/GitLab remote
- Multiple contributors join
- CI/CD pipelines added
- PR reviews needed

### Migration to Remote
When ready:
1. Create GitHub repo
2. Add remote: `git remote add origin <url>`
3. Push with history: `git push -u origin master --tags`
4. Keep `handoffkit/` as repo root
5. Add CI workflows

### Clean History Tips
Before milestone tags, optionally:
```bash
# Interactive rebase to squash fixes
git rebase -i HEAD~5

# Or soft reset and recommit
git reset --soft HEAD~3
git commit -m "feat: complete feature"
```

---
*Last Updated: 2025-09-10*
*Status: Linear development, local-only*