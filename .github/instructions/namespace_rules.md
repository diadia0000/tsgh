# Namespace Coding Style and AI Agent Collaboration Rules

## Coding Philosophy (Linus Torvalds Inspired)
- **Keep it simple**: Avoid over-engineering; prefer straightforward solutions.
- **Readability first**: Code should be written for humans to read, not just for machines to execute.
- **No unnecessary abstractions**: Use abstraction only when it serves clarity and maintainability.
- **Directness**: Don’t hide functionality behind excessive indirection or unnecessary frameworks.
- **Pragmatism**: Choose solutions that work well in practice, not just in theory.

## General Coding Style
- Use consistent indentation (spaces preferred over tabs).
- Functions should be short, single-purpose, and easy to follow.
- Variable names should be descriptive and avoid ambiguity.
- Minimize global state and side effects.
- Code should compile without warnings.

## AI Agent Collaboration Rules
1. **Clarity in Requirements**:  
   Always provide the AI agent with a clear description of the task, including constraints, goals, and expected outputs.

2. **Iterative Development**:  
   Start simple; refine incrementally with AI assistance, following code review principles.

3. **Strict Review Process**:  
   Treat AI-generated code as if it was written by a junior developer—always review, test, and validate.

4. **Namespace Management**:  
   - Each logical component must be placed in a dedicated namespace/module.
   - Avoid namespace pollution by keeping scope minimal and explicit.
   - AI agents must follow predefined naming conventions for namespaces.

5. **Error Handling**:  
   Explicit error checking is mandatory; avoid silent failures.

6. **Logging & Debugging**:  
   - Logging should be centralized.  
   - All modules must use the same logging mechanism.  
   - AI-generated code must not bypass logging rules.

7. **Documentation**:  
   - Every namespace/module must have a clear description of its purpose.  
   - AI agents must auto-generate docstrings and comments where applicable.

8. **Security Awareness**:  
   - No hardcoded credentials or secrets.  
   - AI agents must highlight potential vulnerabilities when generating code.

## Example Namespace Usage (C++-like Pseudocode)
```cpp
namespace network {
    void connect(std::string host, int port);
    void disconnect();
}

namespace database {
    void connect(std::string uri);
    void query(std::string sql);
}
```
