"""Live terminal view for the long autonomous commands (create, climb): what runs now with a bar and ETA, spend against
the budget, overall progress, and the command's own status lines. Log lines scroll above it. Off a terminal (a log
file, a pipe) it prints one status line per stage and one a minute instead. Every call is a no-op without a board."""

import asyncio
import threading
import time
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

BOARD = None
SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'


def dur(s: float | None) -> str:
	if s is None:
		return '--'
	s = int(s)
	return f'{s // 3600}h{s % 3600 // 60:02d}m' if s >= 3600 else f'{s // 60}m{s % 60:02d}s' if s >= 60 else f'{s}s'


def bar(f: float, w: int = 24) -> str:
	f = min(max(f, 0.0), 1.0)
	k = int(f * w * 8)
	return '█' * (k // 8) + ('▏▎▍▌▋▊▉'[k % 8 - 1] if k % 8 and k // 8 < w else '') + '░' * (w - k // 8 - bool(k % 8 and k // 8 < w))


def spark(xs: list[float], lo: float = 0.0, hi: float = 1.0) -> str:
	return ''.join('▁▂▃▄▅▆▇█'[min(7, max(0, int((x - lo) / max(hi - lo, 1e-9) * 7.999)))] for x in xs)


class Board:
	def __init__(self, title: str, budget: float | None = None, spent=lambda: 0.0):
		self.title, self.budget, self.spent = title, budget, spent
		self.t0 = time.monotonic()
		self.stage_ = None  # [label, total, done, t0]
		self.lines: dict[str, Text] = {}
		self.frac = 0.0
		self.last_print = 0.0
		self.live = None

	def stage(self, label: str, total: int) -> None:
		self.stage_ = [label, total, 0, time.monotonic()]
		self.emit(force=True)

	def tick(self, n: int = 1) -> None:
		if self.stage_:
			self.stage_[2] += n
			self.emit(force=self.stage_[2] >= self.stage_[1])

	def stage_eta(self) -> float | None:
		label, total, done, t0 = self.stage_
		return (time.monotonic() - t0) / done * (total - done) if 0 < done < total else 0.0 if done >= total else None

	def eta(self) -> float | None:
		f = max(self.frac, self.spent() / self.budget if self.budget else 0.0)
		return (time.monotonic() - self.t0) * (1 - f) / f if 0.02 < f < 1 else None

	def status(self) -> str:
		parts = [f'[{dur(time.monotonic() - self.t0)}]']
		if self.stage_:
			label, total, done, _ = self.stage_
			parts.append(f'{label} {done}/{total} ETA {dur(self.stage_eta())}')
		if self.budget:
			parts.append(f'${self.spent():.2f}/${self.budget:.0f}')
		if self.frac:
			parts.append(f'overall {self.frac:.0%}' + (f' ETA ~{dur(self.eta())}' if self.frac < 1 else ''))
		return ' | '.join(parts)

	def set_line(self, key: str, text: str | None, markup: bool = False) -> None:
		if text is None:
			self.lines.pop(key, None)
			return
		t = Text.from_markup(text) if markup else Text(text)  # model output and file names stay literal ('[/x]' is not markup)
		if key not in self.lines or self.lines[key].plain != t.plain:
			self.lines[key] = t
			if self.live is None:  # off a terminal a status line is printed once, when it changes
				print(f'  {key}: {t.plain}', flush=True)

	def emit(self, force: bool = False) -> None:
		"""Off a terminal: the progress line when a stage starts or ends, else at most once a minute."""
		if self.live is None and (force or time.monotonic() - self.last_print >= 60):
			self.last_print = time.monotonic()
			print(self.status(), flush=True)

	def __rich__(self):
		t = Table.grid(padding=(0, 1))
		t.add_column(style='bold cyan', no_wrap=True)
		t.add_column(no_wrap=True)
		t.add_column(style='dim', no_wrap=True)
		spin = SPIN[int(time.monotonic() * 8) % len(SPIN)]
		f = max(self.frac, self.spent() / self.budget if self.budget else 0.0)
		t.add_row('overall', Text(bar(f), style='green'), f'{f:4.0%}' + (f'  ETA ~{dur(self.eta())}' if f < 1 else '  done'))
		if self.stage_:
			label, total, done, _ = self.stage_
			t.add_row(f'{spin} {label}' if done < total else f'✓ {label}', Text(bar(done / max(total, 1)), style='yellow'), f'{done}/{total}  ETA {dur(self.stage_eta())}')
		if self.budget:
			s = self.spent()
			t.add_row('budget', Text(bar(s / self.budget), style='red' if s > 0.9 * self.budget else 'magenta'), f'${s:.2f} / ${self.budget:.2f}')
		info = Table.grid(padding=(0, 1))
		info.add_column(style='bold cyan', no_wrap=True)
		info.add_column()
		for k, v in self.lines.items():
			info.add_row(k, v)
		return Panel(
			Group(t, Rule(style='blue'), info) if self.lines else t, title=f'[bold]{self.title}', subtitle=dur(time.monotonic() - self.t0), border_style='blue', expand=False
		)


@contextmanager
def board(title: str, budget: float | None = None, spent=lambda: 0.0, console: Console | None = None):
	"""The view for one command. Nested boards (climb inside a test, create inside create) reuse the outer one."""
	global BOARD
	if BOARD is not None:
		yield BOARD
		return
	b = BOARD = Board(title, budget, spent)
	con = console or Console()
	stop = threading.Event()
	if con.is_terminal:
		b.live = Live(b, console=con, refresh_per_second=4, transient=False, redirect_stdout=True, redirect_stderr=True)
		b.live.start()
	else:

		def beat():  # a long single call (a proposer, a planner) ticks nothing: still say we are alive
			while not stop.wait(60):
				try:
					b.emit()
				except Exception as e:  # a status line must never kill the heartbeat
					print(f'(status unavailable: {type(e).__name__})', flush=True)

		threading.Thread(target=beat, daemon=True).start()
	try:
		yield b
	finally:
		stop.set()
		if b.live is not None:
			b.live.stop()
		else:
			print(b.status(), flush=True)
		BOARD = None


def stage(label: str, total: int) -> None:
	if BOARD:
		BOARD.stage(label, total)


def tick(n: int = 1) -> None:
	if BOARD:
		BOARD.tick(n)


def line(key: str, text: str | None, markup: bool = False) -> None:
	"""A named status line on the board (None removes it)."""
	if BOARD:
		BOARD.set_line(key, text, markup)


def money(spent, budget: float) -> None:
	"""spent: () -> $ so far (prior runs included), shown against the budget and folded into the overall ETA."""
	if BOARD:
		BOARD.spent, BOARD.budget = spent, budget


def progress(f: float) -> None:
	if BOARD:
		BOARD.frac = min(max(f, 0.0), 1.0)


async def gather(coros, label: str | None = None):
	"""asyncio.gather that advances the board as each item finishes. With a label it starts a stage of len(coros)."""
	coros = list(coros)
	if label:
		stage(label, len(coros))

	async def one(c):
		try:
			return await c
		finally:
			tick()

	return await asyncio.gather(*map(one, coros))
