default:
  @just --list

train:
  poetry run python scripts/train.py

inference *args:
  poetry run python scripts/test.py {{args}}