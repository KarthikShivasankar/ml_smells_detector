from ml_code_smell_detector import FrameworkSpecificSmellDetector


def test_framework_detector_initialization():
    detector = FrameworkSpecificSmellDetector()
    assert isinstance(detector, FrameworkSpecificSmellDetector)
    assert len(detector.smells) == 0


def test_generate_report():
    detector = FrameworkSpecificSmellDetector()
    detector.smells = [{'framework': 'Test',
                        'name': 'Test Smell',
                        'how_to_fix': 'Fix it',
                        'benefits': 'Better code',
                        'strategies': 'Do this',
                        'line_number': 1,
                        'code_snippet': 'test code',
                        'file_path': 'test_file.py'}]
    report = detector.generate_report()
    assert 'Test Smell' in report
    assert 'Fix it' in report
    assert 'Better code' in report
    assert 'Do this' in report


def test_import_checker_ignores_from_imports(tmp_py):
    # Regression for #10: names imported from numpy/pandas cannot (and should not) be aliased to np/pd.
    path = tmp_py("""\
        from numpy import ndarray
        from pandas import DataFrame
        import numpy
        import pandas as pd
    """)
    detector = FrameworkSpecificSmellDetector()
    detector.detect_smells(path)
    lines = [s['line_number'] for s in detector.smells if s['name'] == 'Import Checker']
    assert lines == [3]
