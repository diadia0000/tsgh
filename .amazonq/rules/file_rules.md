# AI File Generation Rules

## General Rule

Whenever the generates a file: 1. **Check if the target file already
exists.** 2. If it exists, **delete the old file first**. 3. Then create
the new file with the updated content.

------------------------------------------------------------------------

## Implementation Guidelines

### 1. Python Example

``` python
import os

filename = "output.md"

# If file exists, delete it
if os.path.exists(filename):
    os.remove(filename)

# Write new file
with open(filename, "w", encoding="utf-8") as f:
    f.write("New content here...")
```

### 2. PowerShell Example

``` powershell
$filename = "output.md"

# Remove old file if exists
if (Test-Path $filename) {
    Remove-Item $filename -Force
}

# Create new file
"New content here..." | Out-File -Encoding utf8 $filename
```

### 3. Bash Example

``` bash
filename="output.md"

# Remove old file if exists
[ -f "$filename" ] && rm "$filename"

# Write new file
echo "New content here..." > "$filename"
```

------------------------------------------------------------------------

## Output Rules

When generating files: - Always **overwrite instead of appending**. -
Always **delete old version** before saving new. - Never keep duplicate
versions unless explicitly requested. - If unsure whether to overwrite,
**ask the user**.
