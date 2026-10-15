import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from tools import calculate_expression, get_tool_call_trace, reset_tool_call_guards


class ToolingTests(unittest.TestCase):
    def setUp(self):
        reset_tool_call_guards()

    def test_calculator_returns_expected_value_and_trace(self):
        result = calculate_expression.invoke({"expression": "(5500-1448)/1448*100"})
        self.assertEqual(result, "279.834254")
        trace = get_tool_call_trace()
        self.assertEqual(trace[0]["tool_name"], "calculate_expression")
        self.assertEqual(trace[0]["status"], "completed")

    def test_calculator_rejects_python_code(self):
        result = calculate_expression.invoke({"expression": "__import__('os').getcwd()"})
        self.assertTrue(result.startswith("计算失败："))
        self.assertEqual(get_tool_call_trace()[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
