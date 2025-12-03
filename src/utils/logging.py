"""
Tiny logging helper to keep notebooks clean.
"""

def line():
    print("-" * 80)


def heading(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)