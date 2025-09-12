# C++ Standard Development Rules

## 1. General Principles

-   Always write **valid, compilable, and portable** C++ code.
-   Follow **C++17 or newer standard** (default: C++17 unless explicitly
    stated).
-   Ensure code is **self-contained** with proper `#include` headers.
-   Never assume compiler extensions; stick to standard C++.

------------------------------------------------------------------------

## 2. Code Style

-   **Indentation:** 4 spaces (no tabs).
-   **Naming conventions:**
    -   Classes/Structs: `PascalCase` (e.g., `MyClass`).
    -   Functions: `camelCase` (e.g., `computeResult`).
    -   Variables: `snake_case` (e.g., `max_value`).
    -   Constants: `UPPER_CASE` (e.g., `PI_VALUE`).
-   **Braces:** Always use braces `{}` for control structures, even for
    single-line statements.
-   **Header Guards:** Use `#pragma once` or conventional
    `#ifndef HEADER_H`.

------------------------------------------------------------------------

## 3. Best Practices

-   Prefer **RAII** (Resource Acquisition Is Initialization).
-   Always initialize variables before use.
-   Avoid raw pointers if possible; use `std::unique_ptr`,
    `std::shared_ptr`, or references.
-   Use `const` correctness everywhere applicable.
-   Avoid macros unless necessary; prefer `constexpr` and `inline`
    functions.
-   Use standard containers (`std::vector`, `std::map`, etc.) instead of
    raw arrays when possible.

------------------------------------------------------------------------

## 4. Error Handling

-   Never leave unhandled exceptions.
-   Use `try`/`catch` blocks for exception-prone sections.
-   Prefer **standard exceptions** (`std::runtime_error`,
    `std::invalid_argument`, etc.).
-   For recoverable errors, use `std::optional` or error codes instead
    of throwing.

------------------------------------------------------------------------

## 5. Code Structure

-   Separate **header (.h/.hpp)** and **implementation (.cpp)** files.
-   Keep classes small and focused (Single Responsibility Principle).
-   Avoid long functions (\>50 lines); split into smaller helper
    functions.
-   Document public APIs with brief comments (`///` or `/** */`).

------------------------------------------------------------------------

## 6. Modern C++ Features

-   Use **`auto`** for type inference when it improves readability.
-   Use **range-based for loops** instead of traditional `for` loops
    when possible.
-   Prefer `enum class` over plain `enum`.
-   Use `= default` and `= delete` explicitly for constructors and
    operators when appropriate.
-   Favor `std::thread` and `<thread>` facilities instead of
    platform-specific threading.

------------------------------------------------------------------------

## 7. Testing & Safety

-   Always check boundary conditions when dealing with arrays or
    containers.
-   Avoid undefined behavior (e.g., out-of-bounds access, dangling
    references).
-   Add `assert()` for critical invariants.
-   Write small test programs (`main.cpp`) to verify new components.

------------------------------------------------------------------------

## 8. Example Template

``` cpp
#include <iostream>
#include <vector>
#include <string>

class Example {
public:
    Example() = default;
    ~Example() = default;

    void addValue(int v) {
        values.push_back(v);
    }

    void printValues() const {
        for (const auto& v : values) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

private:
    std::vector<int> values;
};

int main() {
    Example ex;
    ex.addValue(10);
    ex.addValue(20);
    ex.printValues();
    return 0;
}
```

------------------------------------------------------------------------

## 9. Forbidden Practices

❌ Do NOT: - Leave out required `#include`. - Use non-standard compiler
extensions. - Write platform-dependent code unless specified. - Return
raw pointers without clear ownership. - Use `using namespace std;` in
headers.

------------------------------------------------------------------------

## 10. Output Rules for AI

When generating C++ code: 1. Always provide **full code with headers**.
2. Ensure code **compiles without modification**. 3. Follow the style
rules above. 4. If an assumption is required, **state it explicitly in
comments**.
