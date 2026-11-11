"""Publish & Deploy: export, build, copy, and push the public site in one run.

Steps, each stopping the run if it fails:

1. Export: web_public/scripts/export_verified.py writes the verified articles to
   web_public/src/data. It exits non-zero when an article was edited after
   sign-off, so a tampered article stops the deploy here.
2. Build: ``npx astro build --base <base>`` inside web_public.
3. Copy: every file in web_public/dist is copied over the Pages checkout. Files
   are added or replaced, never deleted, and ``.git`` is never touched, because
   the checkout also holds pages that the Astro build does not make.
4. Publish: only the copied paths are staged, committed and pushed, so stray
   files in the checkout (scripts, backups) are never published by accident.
   The run refuses to start if anything is already staged there.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_TARGET: Final = Path.home() / "notitia-civica"
DEFAULT_BASE: Final = "/notitia-civica-ux"
PUBLIC_URL: Final = "https://cls-1000.github.io/notitia-civica-ux/"
STEP_NAMES: Final = ("Export", "Build", "Copy", "Publish")
_GIT_ADD_BATCH: Final = 200

StepState = Literal["pending", "running", "ok", "failed", "skipped"]
CommandRunner = Callable[[Sequence[str], Path], "subprocess.CompletedProcess[str]"]


def run_command(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, capture_output=True, text=True, check=False, timeout=900
    )


@dataclass(frozen=True, slots=True)
class DeployConfig:
    dsn: str
    web_public: Path = REPO_ROOT / "web_public"
    target: Path = DEFAULT_TARGET
    base: str = DEFAULT_BASE
    public_url: str = PUBLIC_URL
    python: str = sys.executable
    push: bool = True


@dataclass(slots=True)
class Step:
    name: str
    state: StepState = "pending"
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DeployRun:
    started_at: str
    steps: list[Step]
    finished_at: str | None = None
    commit: str | None = None
    copied: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.finished_at is None

    @property
    def succeeded(self) -> bool:
        return not self.running and all(step.state == "ok" for step in self.steps)


class DeployError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tail(text: str, limit: int = 40) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


class Deployer:
    """Runs one deploy at a time on a background thread."""

    def __init__(
        self,
        config: DeployConfig,
        *,
        command: CommandRunner = run_command,
        spawn: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.config = config
        self._command = command
        self._spawn = spawn or _spawn_thread
        self._lock = threading.Lock()
        self.current: DeployRun | None = None

    def start(self) -> bool:
        with self._lock:
            if self.current is not None and self.current.running:
                return False
            self.current = DeployRun(_now(), [Step(name) for name in STEP_NAMES])
            run = self.current
        self._spawn(lambda: self._execute(run))
        return True

    # --- Steps ----------------------------------------------------------------

    def _execute(self, run: DeployRun) -> None:
        actions: list[Callable[[Step], None]] = [
            self._export,
            self._build,
            lambda step: self._copy(step, run),
            lambda step: self._publish(step, run),
        ]
        failed = False
        for step, action in zip(run.steps, actions, strict=True):
            if failed:
                step.state = "skipped"
                continue
            step.state = "running"
            try:
                action(step)
            except (DeployError, OSError, subprocess.SubprocessError) as error:
                step.lines.append(f"FAILED: {error}")
                step.state = "failed"
                failed = True
            else:
                if step.state == "running":
                    step.state = "ok"
        run.finished_at = _now()

    def _check(self, step: Step, argv: Sequence[str], cwd: Path) -> str:
        step.lines.append("$ " + " ".join(argv))
        result = self._command(argv, cwd)
        step.lines.extend(_tail(result.stdout))
        step.lines.extend(_tail(result.stderr))
        if result.returncode != 0:
            raise DeployError(f"{argv[0]} exited {result.returncode}")
        return result.stdout

    def _export(self, step: Step) -> None:
        web = self.config.web_public
        self._preflight()
        self._check(
            step,
            [
                self.config.python,
                str(web / "scripts" / "export_verified.py"),
                "--dsn",
                self.config.dsn,
                "--out",
                str(web / "src" / "data"),
            ],
            REPO_ROOT,
        )

    def _build(self, step: Step) -> None:
        self._check(
            step,
            ["npx", "astro", "build", "--base", self.config.base],
            self.config.web_public,
        )

    def _copy(self, step: Step, run: DeployRun) -> None:
        dist = self.config.web_public / "dist"
        target = self.config.target
        if not (dist / "index.html").is_file():
            raise DeployError(f"{dist} has no index.html; the build made no site")
        copied: list[str] = []
        for source in sorted(dist.rglob("*")):
            relative = source.relative_to(dist)
            if relative.parts[0] == ".git":
                raise DeployError("dist contains .git; refusing to copy it")
            if source.is_dir():
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative.as_posix())
        run.copied = copied
        step.lines.append(f"Copied {len(copied)} files into {target} (.git untouched)")

    def _publish(self, step: Step, run: DeployRun) -> None:
        target = self.config.target
        if not self.config.push:
            step.lines.append("Push disabled; files copied but not committed.")
            step.state = "skipped"
            return
        for start in range(0, len(run.copied), _GIT_ADD_BATCH):
            batch = run.copied[start : start + _GIT_ADD_BATCH]
            self._check(step, ["git", "add", "--", *batch], target)
        staged = self._command(["git", "diff", "--cached", "--quiet"], target)
        if staged.returncode == 0:
            step.lines.append("No changes to publish; the live site is current.")
            return
        message = f"deploy(public): publish Astro build from Penfield Desk {_now()}"
        self._check(step, ["git", "commit", "-m", message], target)
        run.commit = self._check(
            step, ["git", "rev-parse", "--short", "HEAD"], target
        ).strip()
        self._check(step, ["git", "push", "origin", "HEAD"], target)

    def _preflight(self) -> None:
        target = self.config.target
        if not (target / ".git").exists():
            raise DeployError(f"{target} is not a git checkout")
        if not (self.config.web_public / "package.json").is_file():
            raise DeployError(f"{self.config.web_public} is not the Astro project")
        if self.config.push:
            staged = self._command(["git", "diff", "--cached", "--quiet"], target)
            if staged.returncode != 0:
                raise DeployError(
                    f"{target} already has staged changes; commit or unstage them "
                    "first so the deploy commit holds only the build"
                )


def _spawn_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True, name="deploy").start()
