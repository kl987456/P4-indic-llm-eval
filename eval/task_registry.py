"""
Task Registry — loads Task definitions from YAML configs and provides
topological ordering for dependency resolution.
"""

from __future__ import annotations

import graphlib
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Task:
    name: str
    language: str
    category: str
    dataset_path: str
    prompt_template: str
    evaluator: str
    metric: str
    few_shot_examples: List[dict] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    # Optional extra fields
    tags: List[str] = field(default_factory=list)
    max_examples: Optional[int] = None
    timeout_seconds: int = 30


class TaskRegistry:
    _tasks: Dict[str, Task] = {}

    @classmethod
    def register(cls, task: Task) -> None:
        """Register a single Task object."""
        if task.name in cls._tasks:
            raise ValueError(f"Task '{task.name}' is already registered.")
        cls._tasks[task.name] = task

    @classmethod
    def register_from_yaml(cls, yaml_path: str) -> Task:
        """
        Parse a YAML file and register the task it describes.
        Returns the created Task.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {yaml_path}")

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        task = Task(
            name=data["name"],
            language=data["language"],
            category=data["category"],
            dataset_path=data["dataset_path"],
            prompt_template=data["prompt_template"],
            evaluator=data["evaluator"],
            metric=data["metric"],
            few_shot_examples=data.get("few_shot_examples", []),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            max_examples=data.get("max_examples"),
            timeout_seconds=data.get("timeout_seconds", 30),
        )
        cls.register(task)
        return task

    @classmethod
    def register_all_from_dir(cls, configs_dir: str) -> List[Task]:
        """Load every *.yaml file in a directory."""
        tasks = []
        for yaml_file in sorted(Path(configs_dir).glob("*.yaml")):
            tasks.append(cls.register_from_yaml(str(yaml_file)))
        return tasks

    @classmethod
    def get(cls, name: str) -> Task:
        if name not in cls._tasks:
            raise KeyError(f"Task '{name}' not found. Registered: {list(cls._tasks)}")
        return cls._tasks[name]

    @classmethod
    def all_tasks(cls) -> Dict[str, Task]:
        return dict(cls._tasks)

    @classmethod
    def topological_order(cls) -> List[str]:
        """
        Topological sort for task dependency ordering (O(V+E)).
        Tasks with no dependencies come first.
        """
        ts = graphlib.TopologicalSorter()
        for name, task in cls._tasks.items():
            ts.add(name, *task.dependencies)
        return list(ts.static_order())

    @classmethod
    def reset(cls) -> None:
        """Clear all registered tasks — useful in tests."""
        cls._tasks.clear()

    @classmethod
    def filter_by_language(cls, language: str) -> List[Task]:
        return [t for t in cls._tasks.values() if t.language == language]

    @classmethod
    def filter_by_category(cls, category: str) -> List[Task]:
        return [t for t in cls._tasks.values() if t.category == category]
