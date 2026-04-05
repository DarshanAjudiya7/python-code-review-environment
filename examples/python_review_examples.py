"""Example Python snippets for exercising the review environment."""

EXAMPLE_SNIPPETS = {
    "unsafe_eval": "\n".join(
        [
            "def load_settings(config_text):",
            "    return eval(config_text)",
        ]
    ),
    "mutable_default": "\n".join(
        [
            "def append_name(name, names=[]):",
            "    names.append(name)",
            "    return names",
        ]
    ),
    "bare_except": "\n".join(
        [
            "def publish_report(report):",
            "    try:",
            '        return report[\"summary\"]',
            "    except:",
            "        return None",
        ]
    ),
    "shell_injection": "\n".join(
        [
            "import subprocess",
            "",
            "def run_script(script_path, user_input):",
            '    cmd = f\"python {script_path} {user_input}\"',
            "    return subprocess.check_output(cmd, shell=True, text=True)",
        ]
    ),
    "syntax_error": "\n".join(
        [
            "def broken_function(",
            "    return 42",
        ]
    ),
    "clean_function": "\n".join(
        [
            "def normalize_name(name: str) -> str:",
            "    cleaned = name.strip().lower()",
            "    return cleaned.replace(\"  \", \" \")",
        ]
    ),
}


EXPECTED_RULE_IDS = {
    "unsafe_eval": {"avoid-eval"},
    "mutable_default": {"mutable-default-list"},
    "bare_except": {"bare-except"},
    "shell_injection": {"shell-true-command-injection"},
    "syntax_error": {"syntax-error"},
    "clean_function": set(),
}
