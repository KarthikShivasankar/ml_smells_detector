"""Tests for Pandas-specific smell detection in FrameworkSpecificSmellDetector."""

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

    def test_only_pandas_receivers_are_reported(self, tmp_py):
        # Regression for #9: nested dict/list access is not chain indexing; pandas reads and writes are.
        code = """\
            import pandas as pd
            config = {"model": {"depth": 3}}
            depth = config["model"]["depth"]
            matrix = [[1, 2], [3, 4]]
            cell = matrix[0][1]
            tags = dict(checks={})
            tags["checks"]["nan"] = "ok"
            df = pd.DataFrame({"a": [1, 2]})
            df["a"][0] = 5
            first = df["a"][0]
            sub = df.copy()
            last = sub["a"][1]
            def f(frame: pd.DataFrame, rows):
                return frame["a"][0], rows["x"]["y"]
        """
        smells = detect(tmp_py, code)
        lines = sorted(s["line_number"] for s in smells if s["name"] == "Chain Indexing")
        assert lines == [9, 10, 12, 14]


class TestChainIndexingNumpyConversions:
    def test_arrays_converted_from_frames_are_not_pandas(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.read_csv('wdbc.csv')
            X = df.loc[:, 2:].values
            y = df.loc[:, 1].to_numpy()
            rows = df.to_dict('records')
            subset = X[y == 1][:40]
            label = rows[0]['label']
            col = df[df['a'] > 0]['b']
        """
        smells = detect(tmp_py, code)
        assert [s["line_number"] for s in smells if s["name"] == "Chain Indexing"] == [8]


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

    def test_ignores_dict_values_and_reports_once(self, tmp_py):
        # Regression for #8: dict.values() was reported, and nested calls reported one .values twice.
        code = """\
            import numpy as np
            import pandas as pd
            counts = {"a": 1, "b": 2}
            arr = np.array(list(counts.values()), dtype=np.int64)
            df = pd.DataFrame({"a": [1, 2]})
            flat = np.asarray(df.values).reshape(-1)
        """
        smells = detect(tmp_py, code)
        lines = [s["line_number"] for s in smells if s["name"] == "DataFrame Conversion Checker"]
        assert lines == [6]


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
    def test_detects_read_without_column_selection(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.read_csv('data.csv')
            val = df['a']
        """
        smells = detect(tmp_py, code)
        assert [s["line_number"] for s in smells if s["name"] == "Column Selection Checker"] == [2]

    def test_no_smell_with_double_bracket(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.read_csv('data.csv')
            subset = df[['a', 'b']]
        """
        smells = detect(tmp_py, code)
        assert "Column Selection Checker" not in names(smells)

    def test_no_smell_with_usecols_or_in_memory_frame(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.read_csv('data.csv', usecols=['a'])
            other = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            val = other['a']
        """
        smells = detect(tmp_py, code)
        assert "Column Selection Checker" not in names(smells)


# ---------------------------------------------------------------------------
# DataFrame Iteration Modification
# ---------------------------------------------------------------------------

class TestDataFrameIterationModification:
    def test_detects_frame_and_row_writes_inside_iterrows(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            for idx, row in df.iterrows():
                df.loc[idx, 'b'] = row['a'] * 2
                row['a'] = 0
        """
        smells = detect(tmp_py, code)
        assert [s["line_number"] for s in smells if s["name"] == "DataFrame Iteration Modification"] == [4, 5]

    def test_no_smell_for_column_loops(self, tmp_py):
        code = """\
            import pandas as pd
            df = pd.DataFrame({'a': ['1'], 'b': ['2']})
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
        """
        smells = detect(tmp_py, code)
        assert "DataFrame Iteration Modification" not in names(smells)


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
