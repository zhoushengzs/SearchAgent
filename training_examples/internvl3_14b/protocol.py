"""Proposed agent template. CPU checked; GPU training remains untested."""

from copy import deepcopy
from llamafactory.data.template import TEMPLATES
from llamafactory.data.formatter import (
    FunctionFormatter,
    ToolFormatter,
    StringFormatter,
)


def register():
    template = deepcopy(TEMPLATES["intern_vl"])
    template.format_tools = ToolFormatter(tool_format="qwen")
    template.format_function = FunctionFormatter(
        slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
    )
    template.format_observation = StringFormatter(
        slots=[
            "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        ]
    )
    TEMPLATES["intern_vl_agent"] = template
    return template
