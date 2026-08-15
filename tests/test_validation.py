import pandas as pd

from src.validation.market_data import MarketDataValidator


VALID_DF = pd.DataFrame(
    [
        {
            "symbol": "INTC",
            "timestamp": "2026-08-01 09:30:00",
            "open": 25.0,
            "high": 26.0,
            "low": 24.5,
            "close": 25.8,
            "volume": 1000,
        },
        {
            "symbol": "INTC",
            "timestamp": "2026-08-02 09:30:00",
            "open": 25.8,
            "high": 26.2,
            "low": 25.5,
            "close": 26.1,
            "volume": 1200,
        },
    ]
)


def test_validate_accepts_valid_market_data():
    validator = MarketDataValidator()

    valid, invalid = validator.validate(VALID_DF.copy())

    assert len(valid) == 2
    assert len(invalid) == 0


def test_validate_rejects_invalid_ohlc_and_duplicate():
    invalid_df = VALID_DF.copy()
    invalid_df.loc[1, "high"] = 20.0
    invalid_df.loc[1, "close"] = 30.0
    invalid_df = pd.concat([invalid_df, invalid_df.iloc[[0]]], ignore_index=True)

    validator = MarketDataValidator()
    valid, invalid = validator.validate(invalid_df)

    assert len(valid) == 1
    assert len(invalid) >= 2
    assert any("OHLC" in str(x) for x in invalid["issues"].tolist())


def test_validate_detects_missing_required_columns():
    bad_df = VALID_DF.drop(columns=["close"])
    validator = MarketDataValidator()

    try:
        validator.validate(bad_df)
        assert False, "Expected schema validation to fail"
    except ValueError:
        pass
