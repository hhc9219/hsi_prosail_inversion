"""
This file defines the DynamicData class. The DynamicData class allows for advanced handling of interdependent data.
It provides a flexible and dynamic way to manage and manipulate data where certain variables depend on others. 

Key features include:
- Dynamic dependency tracking and refreshing when functions are updated.
- Circular dependency detection and error handling.
- Execution of functions with interdependencies, ensuring the correct order based on dependencies.

The class is designed to be extendable and robust, making it suitable for complex data-driven applications.

hhc9219@rit.edu
"""

import json
import pprint
import inspect
import networkx as nx
from typing import Any, Callable
from copy import deepcopy


class DynamicData:

    default_data: dict[str, Any] = {}
    default_exclude_data: bool | list[str] | tuple[str, ...] = True
    default_copy_args: bool = True

    def __init__(self, **data: Any):
        self.exclude_data = self.default_exclude_data
        self.copy_args = self.default_copy_args
        self.deps: dict[str, list[str]] = {}
        self.funcs: dict[str, Callable[..., Any]] = {}
        self.internal_keys: list[str] = list(self.__dict__.keys()) + ["internal_keys"]
        self.__dict__.update(self.default_data)
        self.__dict__.update(data)
        self.refresh_deps()

    def __setattr__(self, name: str, value: Any):
        super().__setattr__(name, value)
        if name == "funcs":
            self.refresh_deps()

    def __getitem__(self, key: str):
        return deepcopy(self.__dict__[key])

    def __setitem__(self, key: str, value: Any):
        if key not in self.internal_keys:
            self.__dict__[key] = value
        else:
            raise KeyError("Internal keys may not be set with __setitem__ .")

    def __call__(self, key: str):
        return deepcopy(self.funcs[key])

    def __str__(self):
        if self.exclude_data is True:
            data = {key: val for key, val in self.__dict__.items() if key not in self.internal_keys}
        elif self.exclude_data is False:
            data = self.__dict__
        else:
            data = {key: val for key, val in self.__dict__.items() if key not in self.exclude_data}

        try:
            result = json.dumps(data, indent=4)
        except TypeError:
            result = pprint.pformat(data, indent=4)
        return result

    def get_data(self):
        if self.exclude_data is True:
            return {key: deepcopy(val) for key, val in self.__dict__.items() if key not in self.internal_keys}
        elif self.exclude_data is False:
            return {key: deepcopy(val) for key, val in self.__dict__.items()}
        else:
            return {key: deepcopy(val) for key, val in self.__dict__.items() if key not in self.exclude_data}

    def get_funcs(self):
        return {key: deepcopy(func) for key, func in self.funcs.items()}

    def get_deps(self):
        return {key: deps.copy() for key, deps in self.deps.items()}

    def update_data(self, **data: Any):
        self.__dict__.update(data)
        if "funcs" in data:
            self.refresh_deps()

    def set_data(self, **data: Any):
        for key in self.__dict__:
            if key not in self.internal_keys:
                del self.__dict__[key]
        self.update_data(**data)

    def update_funcs(self, **funcs: Callable[..., Any]):
        self.funcs.update(funcs)
        self.refresh_deps()

    def set_funcs(self, **funcs: Callable[..., Any]):
        self.funcs.clear()
        self.update_funcs(**funcs)

    def refresh_deps(self):
        self.deps.clear()
        if self.funcs:
            var_deps = {}
            deps_graph = nx.DiGraph()
            for key, func in self.funcs.items():
                var_deps[key] = []
                for dep in inspect.signature(func).parameters:
                    if dep == key:
                        var_deps[key].append(dep)
                    elif dep in self.funcs:
                        deps_graph.add_edge(key, dep)
                    elif dep in self.__dict__:
                        var_deps[key].append(dep)
            if deps_graph.number_of_edges() > 0:
                try:
                    for key in reversed(list(nx.topological_sort(deps_graph))):
                        self.deps[key] = list(deps_graph.neighbors(key))
                    for key in var_deps:
                        if key in self.deps:
                            self.deps[key] += var_deps[key]
                        else:
                            self.deps[key] = var_deps[key]
                except nx.NetworkXUnfeasible:
                    cycles = list(nx.simple_cycles(deps_graph))
                    if cycles:
                        cycle_descriptions = [" -> ".join(cycle) for cycle in cycles]
                        cycle_message = " -> ...\n".join(cycle_descriptions) + " -> ..."
                        error_message = (
                            f"A circular dependency exists in the provided function definitions. The following "
                            f"cycles were found:\n\n{cycle_message}\n\nPlease revise the function dependencies to "
                            "resolve this issue."
                        )
                        raise ValueError(error_message)
            else:
                self.deps.update(var_deps)

    def execute(self, **data: Any):
        for key, deps in self.deps.items():
            if self.copy_args:
                kwargs = {dep: deepcopy(self.__dict__[dep]) for dep in deps}
            else:
                kwargs = {dep: self.__dict__[dep] for dep in deps}
            kwargs.update(data)
            self.__dict__[key] = self.funcs[key](**kwargs)
