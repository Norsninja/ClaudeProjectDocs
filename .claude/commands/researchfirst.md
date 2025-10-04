---
description: Three-tiered research protocol for systematic feature analysis and planning
---
Execute our three-agent research system for: $ARGUMENTS
Always use the  Bash (date '+%Y-%m-%d %H:%M %Z') to learn our correct date and time for this moment. Time needs to be recorded.
As a safeguard this project was Started in September, 2025.

**Research Protocol:**

1. **CODEBASE ANALYSIS** (codebase-researcher agent)
    - Analyze existing patterns, conventions, and architecture
    - Identify exact files and line numbers requiring modification
    - Document current implementation approaches
    - Output: `[topic]_codebase_analysis_[YYYY-MM-DD].md`

2. **EXTERNAL RESEARCH** (technical-research-scout agent)
    - Find production implementations on GitHub
    - Research best practices and proven patterns
    - Identify common pitfalls and solutions
    - Gather performance data if relevant
    - Output: `[topic]_external_research_[YYYY-MM-DD].md`

3. **IMPLEMENTATION PLANNING** (technical-planner agent)
    - Read and synthesize both research documents
    - Create actionable implementation plan with exact integration points
    - Provide code snippets and verification steps
    - Output: `[topic]_implementation_plan_[YYYY-MM-DD].md`

**Execution Requirements:**
- Run agents sequentially, not in parallel
- Each agent must complete before the next begins
- Technical planner MUST read both prior documents
- All outputs save to `project/research/`
- Use consistent topic naming across all three documents

**Final Deliverable:**
After all three documents are complete, provide:
1. Executive summary of key findings
2. Critical decisions requiring user input
3. Recommended implementation approach
4. Any blocking concerns or risks identified

Discuss with Mike what happens after research:
**After Research Completion:**
1. Read all three documents comletely for conflicts
2. Present the paths to all generated documents to the Mike
3. Decide: Implement now, defer, or need more research
4. Update sprint.md with findings if proceeding
5. Create TODO list from implementation plan
6. Identify first concrete action to take


Topic/Feature Details:
$ARGUMENTS