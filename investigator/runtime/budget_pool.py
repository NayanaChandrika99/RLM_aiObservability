# ABOUTME: Tracks shared recursive budget usage across sibling and child loop executions.
# ABOUTME: Provides fair per-sibling budget slices from remaining global budget capacity.

from __future__ import annotations

from dataclasses import dataclass
import time

from investigator.runtime.contracts import RuntimeBudget, RuntimeUsage


def _remaining_int(limit: int, used: int) -> int:
    return max(0, int(limit) - int(used))


def _remaining_float(limit: float, used: float) -> float:
    return max(0.0, float(limit) - float(used))


def _fair_share_int(remaining: int, sibling_count: int) -> int:
    if remaining <= 0:
        return 0
    if sibling_count <= 1:
        return remaining
    share = remaining // sibling_count
    return share if share > 0 else 1


def _fair_share_float(remaining: float, sibling_count: int) -> float:
    if remaining <= 0.0:
        return 0.0
    if sibling_count <= 1:
        return remaining
    return remaining / float(sibling_count)


@dataclass
class BudgetRemaining:
    iterations: int
    depth: int
    tool_calls: int
    subcalls: int
    tokens_total: int | None
    cost_usd: float | None


class RuntimeBudgetPool:
    def __init__(self, *, budget: RuntimeBudget, start_monotonic: float | None = None) -> None:
        self._budget = budget
        self._usage = RuntimeUsage()
        self._subcall_count = 0
        self._started = start_monotonic if start_monotonic is not None else time.monotonic()

    @property
    def budget(self) -> RuntimeBudget:
        return self._budget

    @property
    def usage(self) -> RuntimeUsage:
        return RuntimeUsage(
            iterations=int(self._usage.iterations),
            depth_reached=int(self._usage.depth_reached),
            tool_calls=int(self._usage.tool_calls),
            llm_subcalls=0,
            tokens_in=int(self._usage.tokens_in),
            tokens_out=int(self._usage.tokens_out),
            cost_usd=float(self._usage.cost_usd),
        )

    @property
    def subcall_count(self) -> int:
        return int(self._subcall_count)

    def remaining(self) -> BudgetRemaining:
        tokens_total_used = int(self._usage.tokens_in) + int(self._usage.tokens_out)
        tokens_remaining = None
        if self._budget.max_tokens_total is not None:
            tokens_remaining = _remaining_int(self._budget.max_tokens_total, tokens_total_used)
        cost_remaining = None
        if self._budget.max_cost_usd is not None:
            cost_remaining = _remaining_float(self._budget.max_cost_usd, self._usage.cost_usd)
        return BudgetRemaining(
            iterations=_remaining_int(self._budget.max_iterations, self._usage.iterations),
            depth=_remaining_int(self._budget.max_depth, self._usage.depth_reached),
            tool_calls=_remaining_int(self._budget.max_tool_calls, self._usage.tool_calls),
            subcalls=_remaining_int(self._budget.max_subcalls, self._subcall_count),
            tokens_total=tokens_remaining,
            cost_usd=cost_remaining,
        )

    def allocate_run_budget(self, *, sibling_count: int) -> RuntimeBudget:
        active_siblings = max(1, int(sibling_count))
        remaining = self.remaining()
        return RuntimeBudget(
            max_iterations=_fair_share_int(remaining.iterations, active_siblings),
            max_depth=self._budget.max_depth,
            max_tool_calls=_fair_share_int(remaining.tool_calls, active_siblings),
            max_subcalls=_fair_share_int(remaining.subcalls, active_siblings),
            max_tokens_total=(
                None
                if remaining.tokens_total is None
                else _fair_share_int(remaining.tokens_total, active_siblings)
            ),
            max_cost_usd=(
                None
                if remaining.cost_usd is None
                else _fair_share_float(remaining.cost_usd, active_siblings)
            ),
            sampling_seed=self._budget.sampling_seed,
            max_wall_time_sec=self._budget.max_wall_time_sec,
        )

    def consume(
        self,
        *,
        iterations: int = 0,
        depth: int = 0,
        tool_calls: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        subcalls: int = 0,
    ) -> None:
        self._usage.iterations += max(0, int(iterations))
        self._usage.depth_reached = max(int(self._usage.depth_reached), max(0, int(depth)))
        self._usage.tool_calls += max(0, int(tool_calls))
        self._usage.tokens_in += max(0, int(tokens_in))
        self._usage.tokens_out += max(0, int(tokens_out))
        self._usage.cost_usd += max(0.0, float(cost_usd))
        self._subcall_count += max(0, int(subcalls))

    def budget_reason(self, *, depth: int) -> str | None:
        if depth > self._budget.max_depth:
            return f"max_depth reached: depth={depth} max_depth={self._budget.max_depth}"
        if self._usage.iterations >= self._budget.max_iterations:
            return (
                f"max_iterations reached: iterations={self._usage.iterations} "
                f"max_iterations={self._budget.max_iterations}"
            )
        if self._usage.tool_calls >= self._budget.max_tool_calls:
            return (
                f"max_tool_calls reached: tool_calls={self._usage.tool_calls} "
                f"max_tool_calls={self._budget.max_tool_calls}"
            )
        if self._subcall_count >= self._budget.max_subcalls:
            return (
                f"max_subcalls reached: subcalls={self._subcall_count} "
                f"max_subcalls={self._budget.max_subcalls}"
            )
        if self._budget.max_tokens_total is not None:
            tokens_total = self._usage.tokens_in + self._usage.tokens_out
            if tokens_total >= self._budget.max_tokens_total:
                return (
                    f"max_tokens_total reached: tokens_total={tokens_total} "
                    f"max_tokens_total={self._budget.max_tokens_total}"
                )
        if self._budget.max_cost_usd is not None and self._usage.cost_usd >= self._budget.max_cost_usd:
            return (
                f"max_cost_usd reached: cost_usd={self._usage.cost_usd:.6f} "
                f"max_cost_usd={self._budget.max_cost_usd}"
            )
        elapsed = time.monotonic() - self._started
        if elapsed >= float(self._budget.max_wall_time_sec):
            return (
                f"max_wall_time_sec reached: elapsed={elapsed:.3f} "
                f"max_wall_time_sec={self._budget.max_wall_time_sec}"
            )
        return None
