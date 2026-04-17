# Copy this file to config.local.py for machine-local tweaks.
# config.local.py is ignored by git and is applied after the selected preset.
#
# Presets:
#   $env:LIGHT_LWTI_PRESET = "research_btc_eth_1h"
#
# Direct override file:
#   $env:LIGHT_LWTI_CONFIG = "path/to/another_override.py"

CONFIG_OVERRIDES = {
    # Keep local files separate from committed defaults when needed.
    # "PARQUET_DATASET_VERSION": "local_scratch",

    # Fast local iteration.
    # "SYMBOLS": ["BTC/USDT"],

    # Temporary threshold checks.
    # "ENTRY_THRESHOLD": 0.57,
}

