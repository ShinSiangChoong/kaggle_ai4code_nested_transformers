from src.utils import lr_to_4sf


def test_lr_to_4sf():
    a = [1.23456e-10, 4.32545e-5]
    assert "[1.234e-10, 4.325e-05]" == lr_to_4sf(a)
