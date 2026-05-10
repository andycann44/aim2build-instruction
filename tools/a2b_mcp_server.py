from mcp.server.fastmcp import FastMCP
import subprocess

mcp = FastMCP("Aim2Build")

@mcp.tool()
def git_status() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd="/Users/olly/aim2build-instruction",
        capture_output=True,
        text=True,
    )
    return result.stdout

@mcp.tool()
def compile_instruction_debug() -> str:
    result = subprocess.run(
        [
            "python3",
            "-m",
            "py_compile",
            "clean/routers/instruction_debug.py",
        ],
        cwd="/Users/olly/aim2build-instruction",
        capture_output=True,
        text=True,
    )

    return (
        result.stdout + "\n" + result.stderr
        if result.returncode != 0
        else "compile ok"
    )

if __name__ == "__main__":
    mcp.run()

@mcp.tool()
def compile_all() -> str:
    files = [
        "clean/routers/instruction_debug.py",
        "clean/services/azure_openai_service.py",
        "clean/services/part_candidate_service.py",
    ]

    outputs = []

    for f in files:
        result = subprocess.run(
            ["python3", "-m", "py_compile", f],
            cwd="/Users/olly/aim2build-instruction",
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            outputs.append(f"[OK] {f}")
        else:
            outputs.append(
                f"[FAIL] {f}\n{result.stdout}\n{result.stderr}"
            )

    return "\n\n".join(outputs)