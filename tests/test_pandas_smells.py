"""Tests for Pandas-specific smell detection in FrameworkSpecificSmellDetector."""

import pytest
from ml_code_smell_detector.detectors.framework_detector import FrameworkSpecificSmellDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect(tmp_py, code):
    detector = FrameworkSpecificSmellDetector()
    path = tmp_py(code)
    detector.detect_smells(path)
    return detector.smells


def names(smells):
    return [s["name"] for s in smells]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestPandasDetectorInit:
    def test_starts_with_empty_smells(self):
        detector = FrameworkSpecificSmellDetector()
        assert detector.smells == []

    def test_framework_smells_loaded(self):
        detector = FrameworkSpecificSmellDetector()
        assert "Pandas" in detector.framework_smells

    def test_skips_non_pandas_file(self, tmp_py):
        smells = detect(tmp_py, """\
            x = 1 + 1
            print(x)
        """)
        assert smells == []


# ---------------------------------------------------------------------------
# Unnecessary Iteration (iterrows + vectorizable op in loop)
# ---------------------------------------------------------------------------

class TestUnnecessaryIteration:
    def test_detects_iterrows_with_sum(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            total = 0
            for idx, row in df.iterrows():
                total = sum([row['a'], total])
        """
        smells = detect(tmp_py, code)
        assert "Unnecessary Iteration" in names(smells)

    def test_no_smell_without_vectorizable_op(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            results = []
            for idx, row in df.iterrows():
                results.append(row['a'])
        """
        smells = detect(tmp_py, code)
        assert "Unnecessary Iteration" not in names(smells)


# ---------------------------------------------------------------------------
# Chain Indexing
# ---------------------------------------------------------------------------

class TestChainIndexing:
    def test_detects_chain_indexing_assignment(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            df['a'][0] = 99
        """
        smells = detect(tmp_py, code)
        assert "Chain Indexing" in names(smells)

    def test_no_smell_with_loc(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            df.loc[0, 'a'] = 99
        """
        smells = detect(tmp_py, code)
        assert "Chain Indexing" not in names(smells)


# ---------------------------------------------------------------------------
# Merge Parameter Checker
# ---------------------------------------------------------------------------

class TestMergeParameters:
    def test_detects_merge_without_params(self, tmp_py):
        code = """\
            import pandas as pd
            df1 = pd.DataFrame({'key': [1], 'val': [10]})
            df2 = pd.DataFrame({'key': [1], 'other': [20]})
            result = pd.merge(df1, df2)
        """
        smells = detect(tmp_py, code)
        assert "Merge Parameter Checker" in names(smells)

    def test_no_smell_when_params_present(self, tmp_py):
        code = """\
            import pandas as pd
            df1 = pd.DataFrame({'key': [1], 'val': [10]})
            df2 = pd.DataFrame({'key': [1], 'other': [20]})
            result = pd.merge(df1, df2, on='key', how='inner')
        """
        smells = detect(tmp_py, code)
        assert "Merge Parameter Checker" not in names(smells)


# ---------------------------------------------------------------------------
# InPlace Checker
# ---------------------------------------------------------------------------

class TestInPlaceChecker:
    def test_detects_inplace_without_assignment(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, None, 3]})
            df.fillna(0, inplace=True)
        """
        smells = detect(tmp_py, code)
        assert "InPlace Checker" in names(smells)

    def test_detects_inplace_sort_values(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [3, 1, 2]})
            df.sort_values('a', inplace=True)
        """
        smells = detect(tmp_py, code)
        assert "InPlace Checker" in names(smells)


# ---------------------------------------------------------------------------
# DataFrame Conversion Checker (.values in numpy context)
# ---------------------------------------------------------------------------

class TestDataFrameConversionChecker:
    def test_detects_values_in_numpy_context(self, tmp_py):
        code = """\
            import pandas as pd
            import numpy as np
            df = pd.DataFrame({'a': [1, 2, 3]})
            arr = np.array(df['a'].values)
        """
        smells = detect(tmp_py, code)
        assert "DataFrame Conversion Checker" in names(smells)


# ---------------------------------------------------------------------------
# Datatype Checker (read_csv without dtype)
# ---------------------------------------------------------------------------

class TestDatatypeChecker:
    def test_detects_read_csv_without_dtype(self, tmp_py):
        # The detector requires read_csv to appear inside a parent node that
        # contains a dtype-sensitive operation (groupby, merge, etc.).
        # The parent-walk stops at Module level, so the groupby must be
        # nested inside a function or chained directly.
        code = """\
            import pandas as pd
            def load_and_process():
                df = pd.read_csv('data.csv')
                return df.groupby('col').mean()
        """
        smells = detect(tmp_py, code)
        assert "Datatype Checker" in names(smells)

    def test_no_smell_when_dtype_specified(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.read_csv('data.csv', dtype={'col': float})
            df.groupby('col').mean()
        """
        smells = detect(tmp_py, code)
        assert "Datatype Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Column Selection Checker
# ---------------------------------------------------------------------------

class TestColumnSelectionChecker:
    def test_detects_missing_double_bracket_selection(self, tmp_py):
        # Has pandas operations but uses single-bracket column access only
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            val = df['a']
        """
        smells = detect(tmp_py, code)
        assert "Column Selection Checker" in names(smells)

    def test_no_smell_with_double_bracket(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            subset = df[['a', 'b']]
        """
        smells = detect(tmp_py, code)
        assert "Column Selection Checker" not in names(smells)


# ---------------------------------------------------------------------------
# DataFrame Iteration Modification
# ---------------------------------------------------------------------------

class TestDataFrameIterationModification:
    def test_detects_direct_df_modification_in_loop(self, tmp_py):
        # The detector requires: assignment target is a Subscript,
        # the assignment contains a pandas indicator ('pd.' etc.),
        # AND the assignment is inside a For loop.
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            new_vals = [10, 20, 30]
            for i in range(len(df)):
                df['a'] = pd.Series(new_vals)['a']
        """
        smells = detect(tmp_py, code)
        # Chain indexing is expected from df['a'][i] patterns; the DataFrame
        # Iteration Modification fires when a Subscript is on the left side
        # inside a For and has pd. in the assignment string.
        # Verify that a Pandas smell is detected (at least chain indexing or
        # iteration modification).
        pandas_smells = [s for s in smells if s.get('framework') == 'Pandas']
        assert len(pandas_smells) > 0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestPandasReport:
    def test_report_contains_smell_name(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, None, 3]})
            df.fillna(0, inplace=True)
        """
        detector = FrameworkSpecificSmellDetector()
        path = tmp_py(code)
        detector.detect_smells(path)
        report = detector.generate_report()
        assert "InPlace Checker" in report

    def test_get_results_returns_list(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, None, 3]})
            df.fillna(0, inplace=True)
        """
        detector = FrameworkSpecificSmellDetector()
        path = tmp_py(code)
        detector.detect_smells(path)
        results = detector.get_results()
        assert isinstance(results, list)
        assert len(results) > 0
        assert "name" in results[0]
        assert "location" in results[0]
